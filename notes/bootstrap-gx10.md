# Bootstrap: ASUS Ascent GX10 first boot → fine-tuning ready

Source playbooks (all under `dgx-spark-playbooks/nvidia/`):
- `connect-to-your-spark/` — SSH / NVIDIA Sync
- `dgx-dashboard/` — web dashboard + JupyterLab + system updates
- `pytorch-fine-tune/` — PyTorch container + LoRA/QLoRA scripts
- `vscode/` — IDE on the box or over SSH

This file is updated with the **actual sequence followed on this unit**, including problems hit and how they were resolved. NVIDIA / ASUS official paths are noted where they were the source. If a step looks redundant, it usually exists because skipping it bit us.

---

## Phase 0 — Physical first boot (one-time)

Need keyboard + display attached only for the first boot wizard. After that, fully headless.

1. Plug in 240 W adapter, **10 GbE RJ-45** ethernet (the QSFP ports are for ConnectX-7 pairing, not normal LAN — leave them empty for single-box use).
2. Attach HDMI 2.1 display + USB-C keyboard / mouse.
3. Power on. Follow the on-screen wizard:
   - Region / keyboard layout
   - **Create local user account** (this becomes your SSH login)
   - **Set a password and write it down immediately**. If you forget it and have no second sudo account, the only recovery is a full system reimage (NVIDIA forum confirmation: "security is working as intended"). The recovery image is at ASUS download page for GX10, not the NVIDIA Founders Edition image.
   - Connect to Wi-Fi or confirm wired Ethernet
4. When desktop is up, terminal sanity check:
   ```bash
   uname -m              # aarch64
   hostname              # gx10-XXXX or spark-XXXX
   ip -4 addr            # note IP — Phase 1 will pin it
   nvidia-smi            # NVIDIA GB10, Driver 580.x, CUDA 13.0
   ```

> NVIDIA's setup wizard doc: https://docs.nvidia.com/dgx/dgx-spark/first-boot.html

### If you forget the password

Two cases:

1. **Wrong case / typo** — try variations of what you set. Most "forgot password" reports on the NVIDIA forum end up being keyboard-layout or capitalization mistakes.
2. **Truly forgotten** — there is no recovery shortcut on DGX OS. `sudo passwd` needs sudo. Single-user mode is disabled. You must reimage from the ASUS recovery image (`asus.com/...gx10/helpdesk_download`, ~9 GB, written to USB with Rufus). All data on the 1 TB SSD is wiped.

To make your home life sane after first boot:
```bash
sudo passwd $USER     # root can set any password, even short ones like '123'
                      # pam_pwquality only warns, doesn't enforce, for root path
```
Then immediately store it in a password manager so you never need this again.

---

## Phase 1 — Network: pin a fixed IP, then key-based SSH

mDNS is fragile. Skip it. The home network here is UniFi UXG Fiber; on any router with a DHCP reservation feature the steps are the same.

### Fix the IP on UniFi UXG Fiber

1. UniFi Network UI (`https://unifi.ui.com` or `https://<UXG-IP>`) → **Client Devices**.
2. Find the GX10 (hostname `gx10-XXXX`, MAC vendor NVIDIA / MediaTek / ASUS).
3. **Settings → Network → Fixed IP Address** → assign one (this unit: `192.168.1.200`). Survives reboots, router restarts, lease renewal.
4. SSH by IP from now on: no `.local`, no mDNS, no surprises.

### Key-based SSH (do this before disabling password auth)

On Windows laptop (PowerShell or Git Bash):
```bash
ssh-keygen -t ed25519 -C "hooyao-laptop"
## Default path is fine; passphrase empty for home convenience
```

Push the public key (one-time):
```powershell
## PowerShell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh hooyao@192.168.1.200 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Write `~/.ssh/config` for short commands + auto port-forwarding:
```
Host gx10
    HostName 192.168.1.200
    User hooyao
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 11000 localhost:11000
    LocalForward 8888 localhost:8888
    ServerAliveInterval 60
```

Now `ssh gx10` lands without password and brings Dashboard (11000) + Jupyter (8888) along.

`ServerAliveInterval 60` matters: home routers can drop NAT translations for idle TCP, killing long-running SSH on multi-hour training jobs. 60 s keepalive prevents that.

### Backup the private key

`C:\Users\HuYao\.ssh\id_ed25519` is the only thing standing between you and a manual SSH key reset every laptop reinstall. Copy to OneDrive or an encrypted USB. **Never to GitHub / Slack / public storage.**

---

## Phase 2 — System update

**Lesson learned the hard way: the DGX Dashboard updater is unreliable.** Use the command-line path instead. NVIDIA's own forum recommends this when the Dashboard fails (which it does often — "Update Error Error: 500. Reboot and try again" is a known issue with multiple threads). Command-line path is also documented in `docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html`.

### Standard update sequence

```bash
ssh gx10

sudo apt update
sudo apt -y -o Dpkg::Options::='--force-confdef' -o Dpkg::Options::='--force-confold' dist-upgrade
sudo fwupdmgr refresh --force
sudo fwupdmgr upgrade           # follow [Y/n] prompts; DO NOT lose power during firmware
sudo reboot
```

The `--force-confdef --force-confold` flags keep your local config files when packages ship new defaults. Without them, dpkg interactively prompts you per file and stalls.

### Pitfall A: `thunderbird` snap pre-install script is the slowest part of any DGX OS update

When `apt dist-upgrade` reaches `thunderbird`, the `.deb` is just a stub whose pre-install script downloads the **220 MB Thunderbird snap from Canonical's London CDN** (`canonical-lgw01.cdn.snapcraftcontent.com`). From China that runs at 100-600 KB/s and takes 10-30 minutes, *and the snap download is not resumable*, so a single TCP reset throws away all the progress.

Three ways to survive:

1. **Wait it out.** Fastest path if network is stable.
2. **Configure snapd to use your HTTP proxy** if you have one:
   ```bash
   sudo snap set system proxy.http="http://127.0.0.1:7890"
   sudo snap set system proxy.https="http://127.0.0.1:7890"
   sudo systemctl restart snapd
   ```
3. **Skip thunderbird entirely.** Cleanest path for an LLM workstation since you'll never open it. Note: thunderbird is pulled in by `nvidia-system-station-apps` (a meta-package of GUI app launchers), so you must remove that too:
   ```bash
   sudo dpkg --remove --force-remove-reinstreq thunderbird
   sudo apt remove --purge -y nvidia-system-station-apps 'thunderbird-locale-*'
   sudo apt autoremove --purge -y
   sudo dpkg --configure -a
   ```
   Zero impact on CUDA / driver / Docker / PyTorch / training. You lose a few GUI menu icons you never open.

> **Do NOT just `apt remove thunderbird` alone.** It leaves 18 `thunderbird-locale-*` packages and `nvidia-system-station-apps` in `iU` (unconfigured) state, which breaks every subsequent `apt` operation until you either remove the whole chain or reinstall thunderbird.

### Pitfall B: BFSU / USTC do NOT mirror snap-store

Snap uses a private API protocol (assertions, signing, delta updates), not plain HTTP file mirroring. The advice circulating online to set `SNAP_STORE_PROXY=http://mirrors.bfsu.edu.cn/snap-store` is **wrong**: that URL does not exist, and `SNAP_STORE_PROXY` is not a real snapd configuration variable. The only Canonical-sanctioned "snap mirror" is the Snap Store Proxy enterprise product (license required, PostgreSQL backend, your own domain). Stick with: wait, proxy, or remove.

### Pitfall C: apt locks held by `aptd` after the Dashboard updater dies

If `sudo apt update` reports `Could not get lock /var/lib/apt/lists/lock. It is held by process NNNNN (aptd)`:
```bash
ps -fp NNNNN -o pid,etime,stat,cmd            # check if it's still alive and doing work
sudo systemctl stop packagekit                # the Dashboard backs onto PackageKit -> aptd
sudo kill NNNNN                                # if STAT shows it as S (sleeping) for many minutes
sudo dpkg --configure -a                       # recover any half-configured packages
sudo apt --fix-broken install
```

### Firmware

On this unit, the very first Dashboard update applied all firmware capsules. Subsequent `fwupdmgr` runs report:
```
Devices with the latest available firmware version:
 • Embedded Controller
 • UEFI Device Firmware
 • UEFI Device Firmware
```
No action needed. **DO NOT unplug during firmware upgrades** — multiple reports on the NVIDIA forum of bricked devices from power loss mid-flash.

---

## Phase 3 — Docker permissions + daemon

```bash
sudo usermod -aG docker $USER
exit                 # logout fully, group changes don't apply to existing sessions
ssh gx10             # re-login

docker ps            # should list (likely empty), no permission denied
```

### Pitfall D: `Cannot connect to the Docker daemon` after reboot

If `docker ps` (with or without sudo) returns:
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

Diagnose:
```bash
sudo systemctl status docker.service --no-pager | head -20
sudo journalctl -u docker.service -n 50 --no-pager | grep -iE 'error|fail'
```

On this unit the failure was repeatable: `error initializing buildkit: error creating buildkit instance: invalid database`. dockerd retries 3 times, hits systemd's start-limit, gives up. Root cause: BuildKit's bolt-db state was corrupted by an earlier hard kill of dockerd (probably during the apt-lock mess in Phase 2).

Fix:
```bash
sudo systemctl stop docker.socket
sudo rm -rf /var/lib/docker/buildkit          # daemon rebuilds this on startup
sudo systemctl reset-failed docker.service    # clear the start-limit fuse
sudo systemctl start docker.service
docker ps                                      # should work now
```

This only deletes BuildKit cache. Containers, images, and volumes (`/var/lib/docker/{containers,image,volumes}`) are untouched.

---

## Phase 4 — Verify the container GPU path

NVIDIA Container Toolkit + CDI specs ship pre-installed on DGX OS. Verify with a small CUDA container before pulling the big PyTorch one:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

Expect the same `NVIDIA GB10` info you see on the host. If it errors, check `/var/run/cdi/nvidia.yaml` exists; if missing, `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`.

> **Pull from `nvcr.io`, not Docker Hub.** Same `nvidia/cuda` content, but `auth.docker.io` regularly TLS-handshake-times out from China home networks, while `nvcr.io` (Cloudflare/AWS) is consistently reachable.

---

## Phase 5 — Pull the PyTorch container

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

This is the working environment for SFT / LoRA / QLoRA. **Do NOT install PyTorch on the host.** The NGC container ships:
- NVIDIA-patched PyTorch (Blackwell sm_121 support)
- CUDA 13.0 + cuDNN 9.15 matched to the host driver 580.x
- Transformer Engine (NVFP4 training)
- Flash Attention (precompiled for arm64 — saving you the multi-hour build)
- Apex, Triton, DALI, NCCL, TensorRT

Compressed: ~15 GB download. Extracted: ~20 GB on disk. On this unit's home network with a proxy, the pull averaged 5-30 MB/s with occasional slow layers (single-threaded gzip extract is the final bottleneck, not network).

### Verify

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 python -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.version.cuda)
print('cuDNN   :', torch.backends.cudnn.version())
print('Device  :', torch.cuda.get_device_name(0))
print('SM      :', torch.cuda.get_device_capability(0))     # (12, 1) = sm_121 Blackwell
print('BF16    :', torch.cuda.is_bf16_supported())
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
print('BF16 GEMM OK:', (x @ x.T).shape)
"
```

Expected on this unit:
```
PyTorch : 2.10.0a0+...nv25.11
CUDA    : 13.0
cuDNN   : 91500
Device  : NVIDIA GB10
SM      : (12, 1)
BF16    : True
BF16 GEMM OK: torch.Size([1024, 1024])
```

### Mandatory `docker run` flags for training

The container itself warns on startup. **Use these every time** for real workloads:

```bash
docker run --gpus all -it --rm \
  --ipc=host \                                          # shared memory for DataLoader workers
  --ulimit memlock=-1 \                                 # unlimited pinned memory (CUDA async transfers)
  --ulimit stack=67108864 \                             # 64 MB stack (big CUDA kernels)
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/models:/models:ro \                          # if you pre-downloaded models
  -v $PWD:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
```

Without `--ipc=host`, DataLoader with `num_workers>0` will OOM the default 64 MB `/dev/shm` long before you saturate the GPU. Without `--ulimit memlock=-1`, pinned memory allocations fail at the kernel level and CUDA async copies degrade to sync. Both are footguns that don't error obviously — you just see slow training.

A wrapper script is at `tools/launch_pytorch.sh`.

---

## Phase 6 — Code: edit locally, run on GX10

Two flows. Pick one.

### A. VS Code Remote-SSH (recommended for active development)

1. Install VS Code on laptop, install **Remote - SSH** extension.
2. `F1 → Remote-SSH: Connect to Host... → gx10` (the `~/.ssh/config` alias from Phase 1 just works).
3. Open `/home/hooyao/fine-tuning` on the GX10. Files, terminal, git, debugger — all run server-side on GX10. Your laptop is a thin client.

### B. rsync push, run via SSH (for batch / overnight jobs)

```bash
## from Git Bash on Windows
rsync -avz --delete \
  /c/Users/HuYao/Desktop/fine-tuning/ \
  hooyao@192.168.1.200:/home/hooyao/fine-tuning/
ssh gx10 'cd fine-tuning && bash tools/launch_pytorch.sh python train.py ...'
```

**Don't mix the two flows.** Edits on both sides → merge conflicts. Pick one canonical source.

---

## Phase 7 — First real training

Once Phase 5 + Phase 6 are done and you have models on disk (see `tools/download_models.py`
for the full tier 1-3 set: Llama 3.x / Qwen3 / Gemma 3 dense, plus Qwen3-32B FP8 + Qwen3-30B-A3B MoE;
`notes/curriculum.md` maps each model/dataset to its lesson, method, and memory budget):

```bash
ssh gx10
cd ~/fine-tuning
bash tools/launch_pytorch.sh
## inside container — core stack (always needed):
pip install -U transformers peft datasets trl accelerate
## NF4 QLoRA path (Tier-2 dense) needs bitsandbytes with an aarch64 + sm_121 wheel,
## which is UNVERIFIED on this unit. Tier-3 ships FP8 checkpoints so you can train on
## the native Blackwell path (Transformer Engine) with no bitsandbytes at all.
# pip install bitsandbytes   # try it; if it has no sm_121 build, use the FP8 models instead

## smoke test: load the primary fine-tune base
python -c "from transformers import AutoModelForCausalLM; \
  m = AutoModelForCausalLM.from_pretrained('/models/Qwen/Qwen3-8B'); \
  print(sum(p.numel() for p in m.parameters()))"
```

If that prints ~8.2e9 you're done. (Quick gated-model check: swap in
`/models/meta-llama/Llama-3.2-3B-Instruct` → ~3.2e9.) Move on to actual training scripts.

---

## Quick checklist (after every reboot)

```bash
nvidia-smi                                   # driver alive, no Xid errors
docker ps                                    # daemon up, no-sudo works
df -h /                                      # NVMe headroom — 1 TB total
free -h                                      # 128 GB unified pool, watch swap
```

If `docker ps` fails, see Pitfall D in Phase 3 (BuildKit corruption).

---

## What is NOT in this file

- Multi-Spark Docker Swarm setup (second box): see `dgx-spark-playbooks/nvidia/pytorch-fine-tune/README.md` "Run on two Sparks" and `connect-two-sparks/`.
- Memory-budget arithmetic for picking batch size: separate `notes/memory-budget.md` once we have a real script to compute it.
- DeepSpeed / FSDP configs: separate notes per topic.
- HF model download script for office-network bulk download: `tools/download_models.py` (tier 1-3: Llama 3.x / Qwen3 / Gemma 3 dense, Qwen3 FP8 + MoE, and the SFT/DPO datasets).
