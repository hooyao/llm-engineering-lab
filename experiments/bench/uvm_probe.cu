// GB10 unified-memory functional + bandwidth probe (CUDA C++).
//
// Goal: pin down, at the CUDA-runtime level, how the GB10 (DGX Spark / ASUS
// GX10) unified 128 GB LPDDR5x pool actually behaves — independent of any
// framework. Complements copy_bandwidth.py, which measures the same platform
// from the PyTorch side. Results are written up in notes/hardware-gx10.md,
// "Unified-memory behavior measured on this unit (2026-07-06)".
//
// Build + run (needs the CUDA devel container for nvcc; -arch=sm_121 = GB10):
//     bash experiments/bench/run-uvm-probe.sh
// or manually:
//     nvcc -O3 -arch=sm_121 uvm_probe.cu -o uvm_probe && ./uvm_probe
//
// What it prints:
//   DEVICE + capability flags (unifiedAddressing, managedMemory,
//     concurrentManagedAccess, pageableMemoryAccess,
//     pageableMemoryAccessUsesHostPageTables, directManagedMemAccessFromHost).
//     Note on this unit directManagedMemAccessFromHost reads 0 even though
//     NVIDIA docs say Grace-Blackwell ATS systems should report 1 — see the
//     hardware note for the flagged discrepancy.
//   EXP A: cudaMallocManaged is shared in place (CPU write -> GPU +1 -> CPU
//     read sees it; hostPtr == devicePtr, same virtual address).
//   EXP B: a plain malloc() pointer is dereferenceable INSIDE a kernel with no
//     copy and no managed alloc (ATS / system-allocated memory).
//   EXP C: cudaMemcpy H2D is a real independent copy, never silently aliased to
//     zero-copy — proven by mutating the source after the copy.
//   EXP D: bandwidth — cudaMemcpy H2D/D2H/D2D (copy engines) vs a kernel that
//     streams device memory and a kernel that reads host memory in place over
//     C2C (ATS). The in-place kernel read is the fast path on GB10, not memcpy.
//
// Measured on this unit (2026-07-06, CUDA 13.2 nvcc, driver 580.159.03):
//   H2D pageable 59.3, H2D pinned 58.9, D2H 58.9, D2D 113.2 GB/s;
//   kernel dev<-dev 242.5, kernel dev<-malloc(host) 197.8 GB/s (R+W summed).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA ERROR] %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); exit(1);} }while(0)

__global__ void addOne(float* p, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) p[i]+=1.0f;
}
__global__ void streamCopy(float* __restrict__ dst, const float* __restrict__ src, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) dst[i]=src[i];
}

static double memcpyGBs(void* dst,const void* src,size_t bytes,cudaMemcpyKind k,int it){
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaMemcpy(dst,src,bytes,k)); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for(int i=0;i<it;i++) CK(cudaMemcpy(dst,src,bytes,k));
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
  cudaEventDestroy(a); cudaEventDestroy(b);
  return bytes/(ms/1e3/it)/1e9;
}

int main(){
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
  printf("==== DEVICE ====\n");
  printf("name=%s  cc=%d.%d  SMs=%d\n",p.name,p.major,p.minor,p.multiProcessorCount);
  printf("totalGlobalMem = %.1f GB\n",p.totalGlobalMem/1073741824.0);
  printf("\n==== UNIFIED-MEMORY CAPABILITY FLAGS ====\n");
  printf("unifiedAddressing                      = %d\n",p.unifiedAddressing);
  printf("managedMemory                          = %d\n",p.managedMemory);
  printf("concurrentManagedAccess                = %d\n",p.concurrentManagedAccess);
  printf("pageableMemoryAccess                   = %d\n",p.pageableMemoryAccess);
  printf("pageableMemoryAccessUsesHostPageTables = %d\n",p.pageableMemoryAccessUsesHostPageTables);
  printf("directManagedMemAccessFromHost         = %d\n",p.directManagedMemAccessFromHost);

  size_t N=(size_t)128*1024*1024, bytes=N*sizeof(float);
  int T=256, B=(int)((N+T-1)/T);

  printf("\n==== EXP A: cudaMallocManaged shared CPU<->GPU in place ====\n");
  float* m; CK(cudaMallocManaged(&m,bytes));
  for(size_t i=0;i<N;i++) m[i]=1.0f;
  addOne<<<B,T>>>(m,N); CK(cudaDeviceSynchronize());
  printf("CPU wrote 1.0, GPU +1 -> CPU reads m[0]=%.1f m[N-1]=%.1f (expect 2.0)\n",m[0],m[N-1]);
  cudaPointerAttributes pa; CK(cudaPointerGetAttributes(&pa,m));
  printf("pointerAttr.type=%d (1=host 2=device 3=managed)  device=%p host=%p sameVA=%s\n",
         (int)pa.type,pa.devicePointer,pa.hostPointer,(pa.devicePointer==pa.hostPointer)?"YES":"no");

  printf("\n==== EXP B: plain malloc() pointer dereferenced INSIDE kernel (ATS) ====\n");
  float* h=(float*)malloc(bytes);
  for(size_t i=0;i<N;i++) h[i]=5.0f;
  addOne<<<B,T>>>(h,N);
  cudaError_t eB=cudaDeviceSynchronize();
  printf("kernel on malloc ptr: err=%s  h[0]=%.1f h[N-1]=%.1f (expect 6.0)\n",
         cudaGetErrorString(eB),h[0],h[N-1]);

  printf("\n==== EXP C: is cudaMemcpy H2D alias or real copy? ====\n");
  float* d; CK(cudaMalloc(&d,bytes));
  h[0]=1.0f;
  CK(cudaMemcpy(d,h,bytes,cudaMemcpyHostToDevice));
  h[0]=999.0f;
  float back=0; CK(cudaMemcpy(&back,d,sizeof(float),cudaMemcpyDeviceToHost));
  printf("mutated src to 999 AFTER copy; device holds %.1f -> %s\n",
         back,(back==1.0f)?"INDEPENDENT COPY (NOT zero-copy)":"ALIASED (zero-copy)");

  printf("\n==== EXP D: bandwidth, %.0f MB buffer (peak aggregate ~273 GB/s) ====\n",bytes/1048576.0);
  float* d2; CK(cudaMalloc(&d2,bytes));
  float* hp; CK(cudaMallocHost(&hp,bytes));
  int it=30;
  printf("cudaMemcpy H2D (pageable malloc->dev) = %.1f GB/s\n",memcpyGBs(d,h,bytes,cudaMemcpyHostToDevice,it));
  printf("cudaMemcpy H2D (pinned->dev)          = %.1f GB/s\n",memcpyGBs(d,hp,bytes,cudaMemcpyHostToDevice,it));
  printf("cudaMemcpy D2H (dev->pinned)          = %.1f GB/s\n",memcpyGBs(hp,d,bytes,cudaMemcpyDeviceToHost,it));
  printf("cudaMemcpy D2D (dev->dev)             = %.1f GB/s\n",memcpyGBs(d2,d,bytes,cudaMemcpyDeviceToDevice,it));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto kbw=[&](float* dst,const float* src)->double{
    streamCopy<<<B,T>>>(dst,src,N); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for(int i=0;i<it;i++) streamCopy<<<B,T>>>(dst,src,N);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    return 2.0*bytes/(ms/1e3/it)/1e9;
  };
  printf("kernel streamCopy dev<-dev            = %.1f GB/s (R+W)\n",kbw(d2,d));
  printf("kernel streamCopy dev<-malloc(host)   = %.1f GB/s (R+W, host data over C2C, no memcpy)\n",kbw(d2,h));
  printf("\nDONE\n"); return 0;
}
