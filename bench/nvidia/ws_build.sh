#!/bin/bash
# Box-side: rebuild SPUMA SPDP for one GPU compute capability the runtime bundle does not carry (RTX 5090 and RTX PRO
# 6000 Blackwell are sm_120; the bundle has sm_80/90/100) and drop the result over the unpacked /opt/spumaS, so
# spuma_run.sh and pack_plan.sh run unchanged. NVHPC comes from NVIDIA's apt repo at the bundle's own version (26.9,
# the runtime in /opt/nvrt). Needs spuma_build.sh, spuma_min.sh, boundfix.sh and boundfix/ beside this script.
# Usage: ws_build.sh <source-tree> <work-dir> [cc (120)]     last line is WS-BUILD-DONE or WS-BUILD-FAILED: <step>
set -u
SRC=$1; W=$2; CC=${3:-120}; H=$(cd $(dirname $0) && pwd); export DEBIAN_FRONTEND=noninteractive
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { stamp "WS-BUILD-FAILED: $*"; exit 1; }
if ! ls /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin/nvc++ >/dev/null 2>&1; then
  stamp "nvhpc apt"
  apt-get install -y -qq gnupg curl build-essential flex bison m4 zlib1g-dev libopenmpi-dev rsync >/dev/null 2>&1
  curl -s https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' > /etc/apt/sources.list.d/nvhpc.list
  apt-get update -qq; apt-get install -y -qq ${NVHPC_PKG:-nvhpc-26-9} >/dev/null 2>&1
fi
NV=/opt/nvidia/hpc_sdk; ls $NV/Linux_x86_64/2*/compilers/bin/nvc++ >/dev/null 2>&1 || fail "no nvc++ after apt"
stamp "nvhpc $(ls $NV/Linux_x86_64 | grep '^2' | sort -V | tail -1); build sm_$CC with $(nproc) jobs"
grep -q '^export WM_PRECISION_OPTION=SPDP' $SRC/etc/bashrc || fail "source tree is not SPDP"
L=$SRC/platforms/linux64NvidiaSPDPInt32Opt; mkdir -p $W
for pass in 1 2; do   # the first pass can leave simpleFoam unlinked (cfd2 spdp_build.sh saw it); a second completes it
  mkdir -p $W/pass$pass; NVROOT=$NV LOG=$W/pass$pass NP=$(nproc) bash $H/spuma_build.sh $SRC "$CC"
  [ -x $L/bin/simpleFoam ] && break
done
[ -x $L/bin/simpleFoam ] || fail "no simpleFoam after two passes"
stamp "bound() fix"
NVARCH=$CC bash $H/boundfix.sh $NV $SRC $L/lib $W/bfx || fail boundfix
D=$(ls -d /opt/spumaS/platforms/linux64NvidiaSPDPInt32Opt) || fail "no unpacked bundle under /opt/spumaS"
mv $D $D.bundle && mkdir -p $D && cp -a $L/bin $L/lib $D/ || fail "install"
[ -d $D.bundle/lib/sys-openmpi ] && [ ! -d $D/lib/sys-openmpi ] && cp -a $D.bundle/lib/sys-openmpi $D/lib/
cp $W/bfx/libspumaBoundFix.so $H/libspumaBoundFix_spdp.so || fail "bound() fix copy"   # where pack_plan.sh looks for it
stamp WS-BUILD-DONE
