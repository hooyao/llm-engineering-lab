#!/usr/bin/env python3
"""
Integrity-check downloaded models/datasets and emit a portable SHA256 manifest.

The bytes are verified with SHA256, not just file sizes (sizes only catch
truncation; a flipped bit keeps the same size). HuggingFace records a sha256
per LFS file (weights, large data) even when the transfer used Xet, so we can
prove: bytes-on-disk == bytes-on-Hub.

Modes:
  online (default)  compute sha256 of every local file and compare to the Hub's
                    recorded sha256. Also writes <models-dir>/SHA256SUMS.
  offline (--offline)  verify local files against an existing SHA256SUMS only,
                       no network. Run this on the GX10 after rsync to confirm
                       the transfer didn't corrupt anything.

Layout assumed (what download_models.py writes):  <models-dir>/<org>/<name>/...

Usage:
  python verify_models.py --models-dir G:/hf-cache                  # online, vs Hub
  python verify_models.py --models-dir G:/hf-cache --only Qwen/Qwen3-8B
  python verify_models.py --models-dir /home/hooyao/models --offline  # vs SHA256SUMS

Exit code 0 if everything matches, 1 if any mismatch / missing weight.
On the GX10 you can also use the manifest with coreutils directly:
  cd ~/models && sha256sum -c SHA256SUMS
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

SKIP_DIRS = {".cache", ".git"}      # HF staging + git internals, not payload
BUF = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(BUF), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_repo_files(repo_dir: Path):
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            yield Path(root) / fn


def discover_repos(models_dir: Path) -> list[str]:
    """Return '<org>/<name>' dirs two levels under models_dir."""
    repos = []
    for org in sorted(p for p in models_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        for name in sorted(p for p in org.iterdir() if p.is_dir()):
            repos.append(f"{org.name}/{name.name}")
    return repos


def hf_hashes(repo_id: str, attempts: int = 3):
    """{relpath: (size, sha256_or_None)} from the Hub, or None if unreachable.

    Retries on transient errors so one network blip doesn't leave a repo silently
    unverified. Bails immediately if the repo is genuinely neither model nor dataset.
    """
    import time
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi()
    for attempt in range(attempts):
        not_found = 0
        for getter in (api.model_info, api.dataset_info):
            try:
                info = getter(repo_id, files_metadata=True)
            except RepositoryNotFoundError:
                not_found += 1
                continue
            except Exception:  # noqa: BLE001 - network/auth; retried below
                continue
            out = {}
            for s in info.siblings:
                lfs = getattr(s, "lfs", None)
                sha = (lfs.get("sha256") if isinstance(lfs, dict)
                       else getattr(lfs, "sha256", None)) if lfs else None
                out[s.rfilename] = (s.size, sha)
            return out
        if not_found == 2:
            return None  # genuinely not on the Hub; no point retrying
        if attempt < attempts - 1:
            time.sleep(2)
    return None


def verify_online(models_dir: Path, repos: list[str], manifest_lines: list[str]):
    """Hash each local file, compare to the Hub. Returns (problem_count, skipped_repos)."""
    bad = 0
    skipped: list[str] = []
    for repo in repos:
        repo_dir = models_dir / repo.replace("/", os.sep)
        hf = hf_hashes(repo)
        local = sorted(iter_repo_files(repo_dir))
        n_st = sum(1 for f in local if f.suffix == ".safetensors")
        print(f"--> {repo}  ({len(local)} files, {n_st} safetensors)")
        # Always hash into the manifest, even when the Hub is unreachable.
        digests = {}
        for f in local:
            rel_repo = f.relative_to(repo_dir).as_posix()
            d = sha256_file(f)
            digests[rel_repo] = (f, d)
            manifest_lines.append(f"{d}  {repo}/{rel_repo}")
        if hf is None:
            print("    SKIPPED — Hub unreachable; hashed into manifest but NOT verified vs Hub")
            skipped.append(repo)
            continue
        repo_bad = 0
        for rel_repo, (f, digest) in digests.items():
            if rel_repo not in hf:
                continue  # local extra the Hub omits; not an integrity fault
            size, sha = hf[rel_repo]
            if sha is not None:
                if digest != sha:
                    print(f"    X SHA MISMATCH  {rel_repo}")
                    repo_bad += 1
            elif size is not None and f.stat().st_size != size:
                print(f"    X SIZE MISMATCH  {rel_repo} ({f.stat().st_size} != {size})")
                repo_bad += 1
        # missing-weights check: any Hub safetensors not present locally
        for rel in hf:
            if rel.endswith(".safetensors") and rel not in digests:
                print(f"    X MISSING WEIGHT  {rel}")
                repo_bad += 1
        print(f"    {'OK (verified vs Hub)' if repo_bad == 0 else f'{repo_bad} PROBLEM(S)'}")
        bad += repo_bad
    return bad, skipped


def verify_offline(models_dir: Path, manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        entries.append((digest, rel))
    bad = missing = 0
    for i, (digest, rel) in enumerate(entries, 1):
        f = models_dir / rel
        if not f.exists():
            print(f"    X MISSING  {rel}")
            missing += 1
            continue
        if sha256_file(f) != digest:
            print(f"    X SHA MISMATCH  {rel}")
            bad += 1
        if i % 50 == 0:
            print(f"    ... {i}/{len(entries)} checked")
    print(f"\nchecked {len(entries)} files: {bad} mismatch, {missing} missing")
    return 1 if (bad or missing) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--only", type=str, help="Comma-separated '<org>/<name>' repos to check.")
    ap.add_argument("--offline", action="store_true",
                    help="Verify against an existing SHA256SUMS manifest, no network.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Manifest path (default: <models-dir>/SHA256SUMS).")
    args = ap.parse_args()

    models_dir = args.models_dir.resolve()
    if not models_dir.is_dir():
        print(f"not a directory: {models_dir}", file=sys.stderr)
        sys.exit(2)
    manifest_path = args.manifest or (models_dir / "SHA256SUMS")

    if args.offline:
        print(f"OFFLINE verify of {models_dir} against {manifest_path}\n" + "=" * 60)
        sys.exit(verify_offline(models_dir, manifest_path))

    repos = ([r.strip() for r in args.only.split(",")] if args.only
             else discover_repos(models_dir))
    if not repos:
        print("no <org>/<name> repos found.", file=sys.stderr)
        sys.exit(1)

    print(f"ONLINE verify of {len(repos)} repos under {models_dir} vs HuggingFace\n" + "=" * 60)
    manifest_lines: list[str] = []
    bad, skipped = verify_online(models_dir, repos, manifest_lines)

    # Write the manifest only on a full run (not --only), so it stays authoritative.
    if not args.only:
        manifest_path.write_text("\n".join(sorted(manifest_lines)) + "\n", encoding="utf-8")
        print(f"\nwrote manifest: {manifest_path}  ({len(manifest_lines)} files)")
        print("  on the GX10 after rsync:  cd ~/models && sha256sum -c SHA256SUMS")

    print("\n" + "=" * 60)
    if bad:
        print(f"RESULT: {bad} PROBLEM(S) — investigate the X lines above")
        sys.exit(1)
    if skipped:
        print(f"RESULT: INCOMPLETE — {len(skipped)} repo(s) NOT verified vs Hub (network): {', '.join(skipped)}")
        print(f"  re-run just those:  python tools/verify_models.py --models-dir {models_dir} --only {','.join(skipped)}")
        sys.exit(2)
    print("RESULT: ALL OK — every file verified vs Hub")
    sys.exit(0)


if __name__ == "__main__":
    main()
