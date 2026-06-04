#!/usr/bin/env python3
"""
Download models + datasets for LLM fine-tuning learning curriculum (weeks 1-2).

Designed to run at a fast-network location (office), write to an external drive,
then physically transport home and rsync into the GX10's cache.

Output layout (clean, portable, no HF cache symlink dance):
    <output_dir>/
      meta-llama/Llama-3.2-3B-Instruct/
      Qwen/Qwen3-32B/
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

Gated models (Llama 3.x, Gemma 3): accept license at huggingface.co/<repo_id>
under the same HF account whose token you logged in with. Qwen3 is Apache 2.0
(no gate). Total download budget is kept under ~220 GB to fit a 250 GB drive.
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


# Curriculum (expanded), kept ~180 GB to fit a portable drive with room to spare.
#   Tier 1: small models (<=8B) for full SFT + first LoRA, across 3 architectures.
#   Tier 2: mid models (12-14B) for BF16 LoRA at scale (+ NF4 QLoRA if bnb works).
#   Tier 3: large dense + MoE in FP8 -- Blackwell (sm_121) native low-precision path.
# Model sizes are measured BF16/FP8 safetensors totals (HfApi); only safetensors
# are pulled (legacy *.bin ignored below), so on-disk size ~= the number quoted.
# NOTE on QLoRA: the classic path loads BF16 weights and quantizes to NF4 via
# bitsandbytes. On Blackwell (GB10, sm_121) the native low-precision path is
# NVFP4 / FP8, NOT NF4 -- verify bitsandbytes ships a working aarch64 + sm_121
# wheel before relying on NF4. Tier 3 ships the FP8 checkpoints so you can train
# on the native path regardless. [bitsandbytes/sm_121 status unverified on this unit.]
ASSETS = [
    # ---------------- Tier 1: full SFT + first LoRA (<=8B, 3 architectures) ----------------
    # Sizes are measured BF16 safetensors totals (HfApi files_metadata), rounded up.
    # -- Llama 3.x (GQA + RoPE; gated) --
    Asset("meta-llama/Llama-3.2-1B-Instruct", "model", 2.4, 1, gated=True,
          note="Smallest; full SFT smoke test in minutes. Highest tunability."),
    Asset("meta-llama/Llama-3.2-3B-Instruct", "model", 6.1, 1, gated=True,
          note="Full SFT comfortable on single GB10."),
    Asset("meta-llama/Llama-3.1-8B-Instruct", "model", 15.1, 1, gated=True,
          note="Llama-arch main SFT target."),
    # -- Qwen3 (Apache 2.0, hybrid think/non-think; Apr 2025, supersedes Qwen2.5) --
    Asset("Qwen/Qwen3-1.7B", "model", 3.9, 1,
          note="Tiny Qwen3; fast LoRA iteration, no gate."),
    Asset("Qwen/Qwen3-4B-Instruct-2507", "model", 7.6, 1,
          note="2507 instruct refresh; #2 fine-tune base in 2026 benchmarks."),
    Asset("Qwen/Qwen3-8B", "model", 15.4, 1,
          note="#1 fine-tune base in 2026 benchmarks; replaces Qwen2.5-7B."),
    # -- Gemma 3 (third architecture; gated by Google) --
    Asset("google/gemma-3-4b-it", "model", 8.1, 1, gated=True,
          note="Third arch (bundles a vision tower). Cross-arch LoRA comparison."),

    # ---------------- Tier 2: BF16 LoRA at scale (mid dense) ----------------
    Asset("Qwen/Qwen3-14B", "model", 27.6, 2,
          note="BF16 LoRA sweet spot; also the NF4-QLoRA target if bnb works on sm_121."),
    Asset("google/gemma-3-12b-it", "model", 22.8, 2, gated=True,
          note="Non-Qwen 12B LoRA target; arch diversity at scale."),

    # ---------------- Tier 3: Blackwell native FP8 path (large dense + MoE) ----------------
    Asset("Qwen/Qwen3-32B-FP8", "model", 32.0, 3,
          note="Large dense in FP8 (native sm_121 path); no bitsandbytes needed."),
    Asset("Qwen/Qwen3-30B-A3B-FP8", "model", 30.2, 3,
          note="MoE 30B total / 3B active, FP8; learn MoE fine-tuning."),

    # ---------------- Datasets: SFT (Tier 1) ----------------
    Asset("yahma/alpaca-cleaned", "dataset", 0.05, 1,
          note="52k instructions; classic SFT baseline / smoke test."),
    Asset("databricks/databricks-dolly-15k", "dataset", 0.01, 1,
          note="15k human-written instruction-response pairs."),
    Asset("HuggingFaceTB/smoltalk", "dataset", 4.0, 1,
          note="Current-standard SFT mix (SmolLM); all configs on main."),
    Asset("allenai/tulu-3-sft-mixture", "dataset", 1.4, 1,
          note="939k high-quality SFT pairs across 7 domains (Tulu 3)."),
    Asset("BelleGroup/train_0.5M_CN", "dataset", 0.4, 1,
          note="Chinese instruction tuning, 500k samples."),
    Asset("HuggingFaceH4/ultrachat_200k", "dataset", 1.6, 1,
          note="2023 Zephyr SFT set; kept for A/B vs smoltalk/tulu-3."),

    # ---------------- Datasets: preference / DPO (Tier 2) ----------------
    Asset("HuggingFaceH4/ultrafeedback_binarized", "dataset", 0.7, 2,
          note="DPO reference standard (~63k binarized pairs)."),
    Asset("argilla/dpo-mix-7k", "dataset", 0.005, 2,
          note="Tiny DPO starter (7k); pipeline smoke test."),
    Asset("nvidia/HelpSteer2", "dataset", 0.4, 2,
          note="NVIDIA preference data (helpfulness/correctness attributes)."),
]


# Files we never need; saves bandwidth and disk.
IGNORE_PATTERNS = [
    "*.msgpack",        # Flax weights
    "*.h5",             # TF / Keras weights
    "*.tflite",         # TFLite
    "*.onnx",           # ONNX export
    "*.gguf",           # llama.cpp quantized duplicates (some Gemma/Qwen repos)
    "*.task",           # MediaPipe bundles (Gemma)
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
    ap.add_argument("--tier", type=int, choices=[1, 2, 3],
                    help="Download only tier 1, 2, or 3. Default: all tiers.")
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

3. Copy to GX10. Models live in the hf-cache/ subdir (the drive root may hold
   unrelated files), so sync THAT subdir, not the drive root:

   a) Drive plugged into GX10 directly (mount under /media/hooyao/<LABEL>/):
        rsync -avP /media/hooyao/<LABEL>/hf-cache/  /home/hooyao/models/

   b) Drive on Windows laptop, push over LAN (Git Bash; G: -> /g):
        rsync -avP /g/hf-cache/  hooyao@192.168.1.200:/home/hooyao/models/

   Result layout: /home/hooyao/models/<org>/<model>, e.g. .../Qwen/Qwen3-8B.

4. In your PyTorch training script, load by direct path:
     model = AutoModelForCausalLM.from_pretrained(
         "/home/hooyao/models/Qwen/Qwen3-8B",        # primary fine-tune base (BF16)
         torch_dtype=torch.bfloat16,
     )
     # Tier-3 FP8 (native Blackwell path, no bitsandbytes/NF4):
     #   "/home/hooyao/models/Qwen/Qwen3-32B-FP8"
     dataset = load_dataset(
         "/home/hooyao/models/yahma/alpaca-cleaned", # smoke test; smoltalk/tulu-3 for real SFT
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
