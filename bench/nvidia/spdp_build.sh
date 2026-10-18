#!/bin/bash
# SPUMA in OpenFOAM's mixed precision (WM_PRECISION_OPTION=SPDP: fields and matrices in fp32, linear-solver sums in
# fp64) as a fat sm_80/90/100 bundle unpacking to /opt/spumaS, plus the bound() fix built against it. Runs on any x86 build host from
# a copy of the DP source tree that built the B200 bundle (commit 1ff2d45c). No GPU needed.
# Usage: spdp_build.sh <work-dir>      (log lines go to stdout; the last is SPDP-PACK-DONE or the failing step)
set -u
W=$1; B=$HOME/cfd/baseline; NV=$B/nvhpc/opt; H=$(cd $(dirname $0) && pwd)
mkdir -p $W; cd $W
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
if [ ! -d spumaS ]; then
  stamp "copy source tree"
  rsync -a --exclude platforms --exclude build $B/b200/spumaM/ spumaS/ || { stamp "copy failed"; exit 1; }
fi
sed -i -E 's/^export WM_PRECISION_OPTION=.*/export WM_PRECISION_OPTION=SPDP/' spumaS/etc/bashrc
grep -q '^export WM_PRECISION_OPTION=SPDP' spumaS/etc/bashrc || { stamp "precision not set"; exit 1; }
stamp "build libs + simpleFoam (SPDP, sm 80 90 100)"
NVROOT=$NV LOG=$W NP=${NP:-32} nice -n 10 bash $H/spuma_build.sh $W/spumaS "80 90 100"
L=$W/spumaS/platforms/linux64NvidiaSPDPInt32Opt
[ -x $L/bin/simpleFoam ] || { stamp "first pass left no simpleFoam, second pass"; mkdir -p $W/pass2
  NVROOT=$NV LOG=$W/pass2 NP=${NP:-32} nice -n 10 bash $H/spuma_build.sh $W/spumaS "80 90 100"; }
[ -x $L/bin/simpleFoam ] || { stamp "BUILD FAILED"; exit 1; }
stamp "bound() fix"
NVARCH="80,cc90,cc100" bash $H/boundfix.sh $NV $W/spumaS $L/lib $W/bfx || { stamp "boundfix failed"; exit 1; }
stamp "pack"
PREC=SPDP NAME=spumaS bash $H/spuma_pack.sh $W/spumaS $B/spumaM_rt.tar.zst $W/spumaS_rt.tar.zst "80 90 100" \
  $NV/Linux_x86_64/26.9/cuda/13.3/bin/cuobjdump && stamp SPDP-PACK-DONE
