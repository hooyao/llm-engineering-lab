#!/usr/bin/env python3
"""
Download models + datasets for LLM fine-tuning learning curriculum (weeks 1-2).

Designed to run at a fast-network location (office), write to an external drive,
then physically transport home and rsync into the GX10's cache.

Output layout (clean, portable, no HF cache symlink dance):
    <output_dir>/
      meta-llama/Llama-3.2-3B-Instruct/
      Qwen/Qwen2.5-32B-Instruct/
      ...
      yahma/alpaca-cleaned/         (dataset)

On GX10, load by direct path:
    AutoModelForCausalLM.from_pretrained("/home/hooyao/models/meta-llama/Llama-3.2-3B-Instruct")

Usage:
    pip install -U huggingface_hub hf_transfer
    huggingface-cli login          # for gated models (Llama, Mistral)
    python download_models.py --output-dir E:/hf-cache --tier 1
    python download_models.py --output-dir E:/hf-cache --tier 2
    python download_models.py --output-dir E:/hf-cache --only meta-llama/Llama-3.1-8B-Instruct
    python download_models.py --output-dir E:/hf-cache --dry-run     # just print plan

Resume:
    snapshot_download is resumable file-by-file. Re-run the same command after
    a network failure; finished files are skipped.

Gated models (Llama 3.x, Mistral): accept license at huggingface.co/<repo_id>
under the same HF account whose token you logged in with.
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, GatedRepoError, RepositoryNotFoundError
except ImportError:
    print("Install first:  pip install -U huggingface_hub hf_transfer", file=sys.stderr)
    sys.exit(1)


@dataclass
class Asset:
    repo_id: str
    repo_type: str          # "model" or "dataset"
    size_gb: float          # approximate; for models, BF16 (params * 2 bytes)
    tier: int               # 1 = week 1, 2 = week 2
    gated: bool = False
    note: str = ""


# Curriculum, weeks 1-2.
#   Tier 1: small/medium models for full SFT + first LoRA experiments.
#   Tier 2: mid/large models for LoRA at scale + QLoRA.
# Model size assumes BF16 (params x 2 bytes). QLoRA loads BF16 weights and
# quantizes to NF4 in memory via bitsandbytes; no separate quantized download.
ASSETS = [
    # ---------------- Tier 1 (week 1) ----------------
    Asset("meta-llama/Llama-3.2-1B-Instruct", "model", 2.5, 1, gated=True,
          note="Smallest; full SFT smoke test in minutes."),
    Asset("meta-llama/Llama-3.2-3B-Instruct", "model", 6.0, 1, gated=True,
          note="Full SFT comfortable on single GB10."),
    Asset("meta-llama/Llama-3.1-8B-Instruct", "model", 16.0, 1, gated=True,
          note="Week 1 main target: serious full SFT."),
    Asset("Qwen/Qwen2.5-7B-Instruct", "model", 15.0, 1,
          note="Alternative architecture (RoPE + QKV bias), no gate."),
    Asset("mistralai/Mistral-7B-Instruct-v0.3", "model", 14.0, 1, gated=True,
          note="Third arch: sliding-window attention."),

    # ---------------- Tier 2 (week 2) ----------------
    Asset("Qwen/Qwen2.5-14B-Instruct", "model", 28.0, 2,
          note="LoRA target: BF16 forward + LoRA grads fit."),
    Asset("Qwen/Qwen2.5-32B-Instruct", "model", 64.0, 2,
          note="QLoRA target: 64 GB BF16 -> ~16 GB NF4 in memory."),

    # ---------------- Datasets (all small, useful across weeks) ----------------
    Asset("yahma/alpaca-cleaned", "dataset", 0.05, 1,
          note="52k instructions, classic SFT baseline."),
    Asset("HuggingFaceH4/ultrachat_200k", "dataset", 1.0, 1,
          note="~200k high-quality multi-turn conversations."),
    Asset("BelleGroup/train_0.5M_CN", "dataset", 1.0, 1,
          note="Chinese instruction tuning, 500k samples."),
    Asset("databricks/databricks-dolly-15k", "dataset", 0.01, 1,
          note="15k human-written instruction-response pairs."),
    Asset("argilla/dpo-mix-7k", "dataset", 0.005, 2,
          note="DPO preference pairs, for later weeks."),
]


# Files we never need; saves bandwidth and disk.
IGNORE_PATTERNS = [
    "*.msgpack",        # Flax weights
    "*.h5",             # TF / Keras weights
    "*.tflite",         # TFLite
    "*.onnx",           # ONNX export
    "original/*",       # Llama 3.x ships duplicate .pth + tokenizer.model under original/
    "*.bin",            # legacy pytorch_model*.bin when safetensors exists (re-run without this flag if a model only has .bin)
    "consolidated*.pt", # raw checkpoint duplicates
]


def fmt_size(gb: float) -> str:
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def print_plan(assets: list[Asset]) -> float:
    total = 0.0
    print(f"\n{'Type':<8} {'Repo':<48} {'Size':>10}  Notes")
    print("-" * 115)
    for a in assets:
        marker = " [GATED]" if a.gated else ""
        print(f"{a.repo_type:<8} {a.repo_id:<48} {fmt_size(a.size_gb):>10}  {a.note}{marker}")
        total += a.size_gb
    print("-" * 115)
    print(f"{'TOTAL':<8} {'':<48} {fmt_size(total):>10}")
    return total


def download_one(asset: Asset, output_dir: Path, hf_token: str | None) -> tuple[bool, str]:
    target = output_dir / asset.repo_id
    print(f"\n--> {asset.repo_type:<8} {asset.repo_id}  (~{fmt_size(asset.size_gb)})")
    print(f"    target: {target}")
    try:
        snapshot_download(
            repo_id=asset.repo_id,
            repo_type=asset.repo_type,
            local_dir=str(target),
            token=hf_token,
            ignore_patterns=IGNORE_PATTERNS,
            max_workers=8,
        )
        return True, ""
    except GatedRepoError:
        msg = f"GATED: accept license at https://huggingface.co/{asset.repo_id}"
        print(f"    X {msg}")
        return False, msg
    except RepositoryNotFoundError:
        msg = "REPO NOT FOUND (renamed? typo?)"
        print(f"    X {msg}")
        return False, msg
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", "?")
        msg = f"HTTP {status}: {e}"
        if status == 401:
            msg = "AUTH FAILED. Run: huggingface-cli login   (or set HF_TOKEN)"
        print(f"    X {msg}")
        return False, msg
    except KeyboardInterrupt:
        print("\n    ! interrupted; re-run same command to resume.")
        raise
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"    X {msg}")
        return False, msg


def actual_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024**3


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Root directory. e.g. E:/hf-cache (Windows) or /mnt/drive/hf-cache (Linux)")
    ap.add_argument("--tier", type=int, choices=[1, 2],
                    help="Download only tier 1 (week 1) or tier 2 (week 2). Default: both.")
    ap.add_argument("--only", type=str,
                    help="Comma-separated repo IDs to download (overrides --tier).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + free-space check, then exit.")
    ap.add_argument("--no-fast-transfer", action="store_true",
                    help="Disable hf_transfer (use pure-Python downloader).")
    args = ap.parse_args()

    # Filter assets.
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        assets = [a for a in ASSETS if a.repo_id in wanted]
        missing = wanted - {a.repo_id for a in assets}
        if missing:
            print(f"Unknown repo IDs: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
    elif args.tier:
        assets = [a for a in ASSETS if a.tier == args.tier]
    else:
        assets = list(ASSETS)

    if not assets:
        print("No assets selected.", file=sys.stderr)
        sys.exit(1)

    total_gb = print_plan(assets)

    # Free-space check.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.output_dir).free / 1024**3
    print(f"\nFree at {args.output_dir}: {free_gb:.1f} GB   (need ~{total_gb * 1.1:.0f} GB with 10% slack)")
    if free_gb < total_gb * 1.1:
        print("  ! Not enough free space. Free up some, or use a different drive.")
        sys.exit(1)

    if args.dry_run:
        print("\n(dry-run; nothing downloaded)")
        return

    # Speed up with hf_transfer (Rust-based, parallel chunks). Big win on fast links.
    if not args.no_fast_transfer:
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("\n[+] hf_transfer enabled (parallel chunked downloader)")
        except ImportError:
            print("\n[!] hf_transfer not installed; downloads will be slower.")
            print("    pip install hf_transfer    (recommended on fast networks)")

    hf_token = os.environ.get("HF_TOKEN")  # falls back to huggingface-cli login token

    print(f"\nDownloading to: {args.output_dir}\n" + "=" * 70)
    results: list[tuple[Asset, bool, str]] = []
    for a in assets:
        ok, msg = download_one(a, args.output_dir, hf_token)
        results.append((a, ok, msg))

    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    ok_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - ok_count
    print(f"  OK:     {ok_count}/{len(results)}")
    if fail_count:
        print(f"  FAILED: {fail_count}")
        for a, ok, msg in results:
            if not ok:
                print(f"    - {a.repo_id}: {msg}")

    # Actual on-disk sizes.
    print(f"\nActual sizes under {args.output_dir}:")
    total_actual = 0.0
    for a in assets:
        gb = actual_size_gb(args.output_dir / a.repo_id)
        total_actual += gb
        print(f"  {fmt_size(gb):>10}  {a.repo_id}")
    print(f"  {'-' * 10}")
    print(f"  {fmt_size(total_actual):>10}  TOTAL")

    # Copy-back instructions.
    print(f"""
---------------------------------------------------------------------
NEXT STEPS

1. Verify total size:
     Linux:    du -sh {args.output_dir}
     Windows:  Right-click drive -> Properties

2. Unplug drive. At home, plug into GX10 (USB-C) or into your laptop.

3. Copy to GX10. Two options:

   a) Drive plugged into GX10 directly (mount appears under /media/hooyao/...):
        rsync -avP /media/hooyao/<DRIVE>/  /home/hooyao/models/

   b) Drive on Windows laptop, push over LAN:
        scp -r /e/hf-cache/  hooyao@192.168.1.200:/home/hooyao/models/
      or rsync via WSL / Git Bash:
        rsync -avP /e/hf-cache/  hooyao@192.168.1.200:/home/hooyao/models/

4. In your PyTorch training script, load by direct path:
     model = AutoModelForCausalLM.from_pretrained(
         "/home/hooyao/models/meta-llama/Llama-3.2-3B-Instruct",
         torch_dtype=torch.bfloat16,
     )
     dataset = load_dataset(
         "/home/hooyao/models/yahma/alpaca-cleaned",
     )

5. Inside the Docker container, bind-mount the directory:
     docker run --gpus all -it --rm --ipc=host \\
       -v /home/hooyao/models:/models:ro \\
       -v $HOME/.cache/huggingface:/root/.cache/huggingface \\
       -v $PWD:/workspace -w /workspace \\
       nvcr.io/nvidia/pytorch:25.11-py3
   Then load with /models/<org>/<name>.
---------------------------------------------------------------------
""")


if __name__ == "__main__":
    main()
