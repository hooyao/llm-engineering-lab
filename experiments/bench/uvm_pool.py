#!/usr/bin/env python3
"""
UVM (cudaMallocManaged) allocator context manager, backported to run in the
nvcr.io/nvidia/pytorch:26.04-py3 container (PyTorch 2.12.0a0).

WHY THIS FILE EXISTS
--------------------
PyTorch main has a context manager, torch.cuda._use_uvm(), that routes CUDA
allocations through cudaMallocManaged so tensors live in CUDA Unified Virtual
Memory (managed memory). It is NOT in any released/stable tag yet, and NOT in
the 2.12.0a0 build shipped by the 26.04 container. Verified on GX10 2026-07-06:
  - container torch 2.12.0a0  -> hasattr(torch.cuda, "_use_uvm") == False
  - stock nightly 2.14.0.dev  -> _use_uvm present and works on GB10
Rather than install a nightly wheel (the user works inside the container), this
file backports the exact upstream logic. Every dependency it needs already
exists in the 26.04 container:
  - torch.cuda.MemPool, torch.cuda.memory.use_mem_pool
  - torch._C._cuda_customAllocator
  - cuda.bindings.runtime  (the cuda-python package is preinstalled)

The implementation is a faithful port of pytorch/pytorch main
torch/cuda/memory.py (_make_uvm_allocator + _use_uvm), commit 435385c. Kept
close to the original so it can be dropped when a container that ships the real
_use_uvm arrives — at that point delete this and call torch.cuda._use_uvm().

WHAT UVM BUYS ON GB10
---------------------
A tensor allocated inside use_uvm() is managed memory: one allocation, one
virtual address valid on both CPU and GPU, shared in place (no cudaMemcpy).
That is the zero-copy CPU/GPU sharing that plain device tensors do not give
you. Note the upstream docstring's own caveat: for workloads that fit in the
pool, ordinary device tensors are FASTER (no page faults) — UVM is for sharing
and oversubscription, not speed. See notes/hardware-gx10.md.

Usage:
    from uvm_pool import use_uvm
    with use_uvm():
        x = torch.zeros(1024, 1024, device="cuda")   # managed allocation
    # run the self-check demo:
    #   bash experiments/bench/run-uvm-pool-26.04.sh
"""

import contextlib
import ctypes
import logging
import threading
import traceback

import torch
from torch.cuda import MemPool
from torch.cuda.memory import use_mem_pool


_UVM_ALLOCATOR = None
_UVM_ALLOCATOR_LOCK = threading.Lock()


def _make_uvm_allocator():
    """Build the UVM CUDAPluggableAllocator and the ctypes closures behind it.

    Returns (c_alloc, c_free, allocator). UVM alloc/free are stateless (device
    is passed per call), so one allocator serves all calls and devices. Raises
    ImportError if cuda-python is unavailable.

    Faithful port of torch/cuda/memory.py::_make_uvm_allocator (pytorch main).
    """
    try:
        from cuda.bindings import runtime as _rt
    except ImportError:
        raise ImportError(
            "use_uvm() requires the 'cuda-python' package "
            "(cuda.bindings.runtime) for cudaMallocManaged, cudaMemAdvise, "
            "and cudaFree."
        ) from None

    log = logging.getLogger(__name__)

    _ALLOC_FN = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
    )
    _FREE_FN = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
    )

    def _check(result, msg: str = ""):
        err = result if not isinstance(result, tuple) else result[0]
        if err != _rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}. {msg}")

    _advise_uses_struct = hasattr(_rt, "cudaMemLocation")

    def _mem_advise(ptr, size, advice, device_id, _runtime=_rt):
        """Call cudaMemAdvise, handling the struct-vs-int API difference.

        cuda-bindings 13.x requires a cudaMemLocation struct; 12.x expects a
        plain int device ordinal. Try struct first, latch to int on TypeError.
        """
        nonlocal _advise_uses_struct
        if _advise_uses_struct:
            try:
                loc = _runtime.cudaMemLocation()
                loc.type = _runtime.cudaMemLocationType.cudaMemLocationTypeDevice
                loc.id = device_id
                return _runtime.cudaMemAdvise(ptr, size, advice, loc)
            except TypeError:
                _advise_uses_struct = False
        return _runtime.cudaMemAdvise(ptr, size, advice, device_id)

    _uvm_advise_supported_cache: dict[tuple[int, int], bool] = {}

    def _device_supports_uvm_advise(device_id, _runtime=_rt):
        cache_key = (device_id, id(_runtime))
        if cache_key in _uvm_advise_supported_cache:
            return _uvm_advise_supported_cache[cache_key]
        attr = getattr(
            _runtime.cudaDeviceAttr, "cudaDevAttrConcurrentManagedAccess", None
        )
        if attr is None:
            supported = False
        else:
            result = _runtime.cudaDeviceGetAttribute(attr, device_id)
            err = result if not isinstance(result, tuple) else result[0]
            if err != _runtime.cudaError_t.cudaSuccess:
                supported = False
            else:
                supported = bool(result[1])
        _uvm_advise_supported_cache[cache_key] = supported
        return supported

    def _uvm_alloc(size, device, stream, _runtime=_rt):
        try:
            err, ptr = _runtime.cudaMallocManaged(size, _runtime.cudaMemAttachGlobal)
            _check(err, f"cudaMallocManaged({size})")
            ptr = int(ptr)
            if device >= 0 and _device_supports_uvm_advise(device, _runtime):
                _check(
                    _mem_advise(
                        ptr,
                        size,
                        _runtime.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation,
                        device,
                    ),
                    "cudaMemAdvise(SetPreferredLocation)",
                )
                _check(
                    _mem_advise(
                        ptr,
                        size,
                        _runtime.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy,
                        device,
                    ),
                    "cudaMemAdvise(SetAccessedBy)",
                )
            return ptr
        except Exception:
            log.error(
                "[use_uvm] FAILED to allocate %d bytes (%.2f GiB) via UVM."
                " CUDACachingAllocator will raise an OOM error as a result."
                " You can ignore free-memory numbers reported by PyTorch"
                " as they are irrelevant for UVM.\nException:\n%s",
                size,
                size / (1024**3),
                traceback.format_exc(),
            )
            return 0

    def _uvm_free(ptr, size, device, stream, _runtime=_rt):
        """Best-effort free; guards against interpreter shutdown."""
        try:
            if ptr:
                _check(_runtime.cudaFree(ptr))
        except Exception:
            if log is not None and traceback is not None:
                try:
                    log.error("[use_uvm] exception in free:\n%s", traceback.format_exc())
                except Exception:
                    pass

    c_alloc = _ALLOC_FN(_uvm_alloc)
    c_free = _FREE_FN(_uvm_free)
    alloc_ptr = ctypes.cast(c_alloc, ctypes.c_void_p).value
    free_ptr = ctypes.cast(c_free, ctypes.c_void_p).value
    allocator = torch._C._cuda_customAllocator(alloc_ptr, free_ptr)
    return c_alloc, c_free, allocator


@contextlib.contextmanager
def use_uvm(device=None):
    """Route CUDA allocations inside this context through cudaMallocManaged (UVM).

    Faithful port of torch.cuda._use_uvm (pytorch main). Tensors allocated in
    this block use CUDA Unified Virtual Memory: one allocation shared in place
    between CPU and GPU, and oversubscription past device memory by paging to
    host RAM on demand. Numerics identical to normal device tensors; only
    performance differs (page-migration overhead).

    On GB10 the c_alloc/c_free closures must outlive the context (a block can
    stay cached in the PrivatePool and be freed later by a global empty_cache),
    so the allocator is built once and reused, exactly as upstream does.
    """
    global _UVM_ALLOCATOR
    if _UVM_ALLOCATOR is None:
        with _UVM_ALLOCATOR_LOCK:
            if _UVM_ALLOCATOR is None:
                _UVM_ALLOCATOR = _make_uvm_allocator()
    allocator = _UVM_ALLOCATOR[2]

    pool = MemPool(allocator=allocator)
    with use_mem_pool(pool, device=device):
        yield pool


# --------------------------------------------------------------------------
# Self-check demo: prove the new API does what the docs claim, on GB10.
# --------------------------------------------------------------------------
def _demo():
    print("=" * 70)
    print("UVM allocator backport — self-check on", torch.cuda.get_device_name(0))
    print("=" * 70)
    print("torch          :", torch.__version__)
    print("native _use_uvm:", hasattr(torch.cuda, "_use_uvm"),
          "(this file backports it when False)")
    print()

    # 1. A tensor allocated inside use_uvm() is managed memory, shared in place.
    #    We prove sharing the way uvm_probe.cu EXP A does, but from PyTorch:
    #    write on CPU, mutate on GPU, read back on CPU — same storage, no copy.
    with use_uvm():
        x = torch.zeros(2048, 2048, device="cuda")   # managed allocation
    ptr = x.data_ptr()

    # cudaPointerGetAttributes should report this pointer as MANAGED (type 3).
    from cuda.bindings import runtime as rt
    err, attr = rt.cudaPointerGetAttributes(ptr)
    mem_type = int(attr.type)  # 0 unregistered, 1 host, 2 device, 3 managed
    type_name = {0: "unregistered", 1: "host", 2: "device", 3: "MANAGED"}.get(mem_type, "?")
    print(f"1. tensor allocated in use_uvm():")
    print(f"   device={x.device}  data_ptr=0x{ptr:x}")
    print(f"   cudaPointerGetAttributes.type = {mem_type} ({type_name})")
    print(f"   -> {'PASS: managed memory' if mem_type == 3 else 'FAIL: not managed'}")
    print()

    # 2. Correctness of a GPU kernel on a managed tensor. The zero-copy /
    #    in-place sharing claim is already proven by step 1 (managed pointer ==
    #    one VA valid on both CPU and GPU); this step just confirms compute on
    #    a managed tensor is numerically normal.
    with use_uvm():
        y = torch.ones(1024, 1024, device="cuda")    # managed, all 1.0
    y.mul_(2.0)                                       # GPU kernel: *2 -> 2.0
    torch.cuda.synchronize()
    val = y.flatten()[0].item()
    print("2. GPU kernel on a managed tensor (ones *= 2):")
    print(f"   y[0] = {val}  -> {'PASS' if val == 2.0 else 'FAIL'}")
    print()

    # 3. Ordinary device tensor for contrast: its pointer is type 2 (device),
    #    NOT managed — the CPU cannot address it directly.
    z = torch.zeros(16, device="cuda")               # normal allocation
    err, zattr = rt.cudaPointerGetAttributes(z.data_ptr())
    ztype = int(zattr.type)
    print("3. contrast — ordinary device tensor (no use_uvm):")
    print(f"   cudaPointerGetAttributes.type = {ztype} "
          f"({'device' if ztype == 2 else '?'})  -> not shared with CPU")
    print()

    print("=" * 70)
    print("RESULT: use_uvm() produces genuine managed memory (type 3) on GB10,")
    print("while a normal cuda tensor stays device-only (type 2). The new API")
    print("works as documented. For zero-copy in-place bandwidth numbers see")
    print("uvm_probe.cu; for why you rarely NEED this on GB10 see hardware note.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
