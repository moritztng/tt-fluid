#!/bin/bash
# Device memory bandwidth of one idle GPU, measured: a device-local copy and a triad over 8 GiB fp32 arrays (PyTorch).
# Counted the STREAM way: copy moves 2 arrays (one read, one written), triad 3. Run it on an idle GPU only.
# Usage: bw_probe.sh prep        installs PyTorch into /root/bwenv (minutes, network and disk only)
#        bw_probe.sh run [gpu]   prints one BW line per kernel (seconds); needs prep
set -u
V=/root/bwenv
if [ "$1" = prep ]; then
  export DEBIAN_FRONTEND=noninteractive
  python3 -m venv $V 2>/dev/null || { apt-get install -y -qq python3-venv >/dev/null 2>&1; python3 -m venv $V; } || { echo "BW-PREP-FAILED: venv"; exit 1; }
  $V/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126 && echo BW-PREP-DONE || echo "BW-PREP-FAILED: pip"
  exit
fi
CUDA_VISIBLE_DEVICES=${2:-0} $V/bin/python - <<'PY'
import time, torch
n = 2 ** 31                                  # 8 GiB per fp32 array
a, b, c = (torch.empty(n, device="cuda") for _ in range(3))
b.fill_(1.0); c.fill_(2.0)
def rate(f, arrays, reps=30):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    return arrays * 4 * n * reps / (time.perf_counter() - t) / 1e9
name = torch.cuda.get_device_name(0)
print(f"BW copy  {rate(lambda: a.copy_(b), 2):.0f} GB/s  ({name}, 8 GiB fp32 arrays, 30 reps)")
print(f"BW triad {rate(lambda: torch.add(b, c, alpha=3.0, out=a), 3):.0f} GB/s")
assert float(a[12345]) == 7.0
PY
