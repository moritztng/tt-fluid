#!/bin/bash
# Box-side: rebuild SPUMA SPDP for one GPU compute capability the runtime bundle does not carry (RTX 5090 and RTX PRO
# 6000 Blackwell are sm_120; the bundle has sm_80/90/100) and drop the result over the unpacked /opt/spumaS, so
# spuma_run.sh and pack_plan.sh run unchanged. NVHPC comes from NVIDIA's apt repo at the bundle's own version (26.9,
# the runtime in /opt/nvrt). Needs spuma_build.sh, spuma_min.sh, boundfix.sh and boundfix/ beside this script.
# Usage: ws_build.sh build <source-tree> <work-dir> [cc (120)]   builds, writes <work-dir>/spumaS_sm<cc>.tar.zst, installs it
#        ws_build.sh install <tarball>                            installs a tarball an earlier build wrote (no compiler needed)
# The last line is WS-BUILD-DONE or WS-BUILD-FAILED: <step>. The build is about an hour of 32 jobs (cfd2: 4,158 s).
set -u
H=$(cd $(dirname $0) && pwd); export DEBIAN_FRONTEND=noninteractive
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { stamp "WS-BUILD-FAILED: $*"; exit 1; }
install() {   # platform bin + lib over the unpacked bundle (its sys-openmpi kept), bound() fix where pack_plan.sh looks
  local D; D=$(ls -d /opt/spumaS/platforms/linux64NvidiaSPDPInt32Opt) || fail "no unpacked bundle under /opt/spumaS"
  [ -d $D.bundle ] || mv $D $D.bundle; rm -rf $D; mkdir -p $D
  tar xf $1 -I zstd -C $D || fail "install: unpack $1"
  mv $D/libspumaBoundFix_spdp.so $H/ || fail "install: bound() fix"
  [ -d $D.bundle/lib/sys-openmpi ] && [ ! -d $D/lib/sys-openmpi ] && cp -a $D.bundle/lib/sys-openmpi $D/lib/
  [ -x $D/bin/simpleFoam ] || fail "install: no simpleFoam"
  stamp WS-BUILD-DONE
}
jobs_fit() {   # compile jobs the container can carry: its cgroup CPU quota (nproc reports the host's threads), 2 GB each
  local c m x; c=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); [ "${c:-0}" -ge 1 ] || c=$(nproc)
  m=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo); x=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  [ "${x:-max}" != max ] && [ $x -lt $m ] && m=$x
  m=$(( m / 2147483648 )); [ $m -lt $c ] && c=$m; [ $c -ge 1 ] || c=1; echo $c
}
[ "${1:-}" = install ] && { install $2; exit 0; }
[ "${1:-}" = build ] || fail "usage: build <source-tree> <work-dir> [cc] | install <tarball>"
SRC=$2; W=$3; CC=${4:-120}
if ! ls /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin/nvc++ >/dev/null 2>&1; then
  stamp "nvhpc apt"   # nvhpc-26-9 is a 5 GB download (NVIDIA's apt index, 2026-11-28)
  apt-get install -y -qq gnupg curl build-essential flex bison m4 zlib1g-dev libopenmpi-dev rsync >/dev/null 2>&1
  curl -s https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' > /etc/apt/sources.list.d/nvhpc.list
  apt-get update -qq; apt-get install -y -qq ${NVHPC_PKG:-nvhpc-26-9} >/dev/null 2>&1
fi
NV=/opt/nvidia/hpc_sdk; ls $NV/Linux_x86_64/2*/compilers/bin/nvc++ >/dev/null 2>&1 || fail "no nvc++ after apt"
NP=$(jobs_fit); stamp "nvhpc $(ls $NV/Linux_x86_64 | grep '^2' | sort -V | tail -1); build sm_$CC with $NP jobs"
grep -q '^export WM_PRECISION_OPTION=SPDP' $SRC/etc/bashrc || fail "source tree is not SPDP"
L=$SRC/platforms/linux64NvidiaSPDPInt32Opt; mkdir -p $W
for pass in 1 2; do   # the first pass can leave simpleFoam unlinked (cfd2 spdp_build.sh saw it); a second completes it
  mkdir -p $W/pass$pass; NVROOT=$NV LOG=$W/pass$pass NP=$NP bash $H/spuma_build.sh $SRC "$CC"
  [ -x $L/bin/simpleFoam ] && break
done
[ -x $L/bin/simpleFoam ] || fail "no simpleFoam after two passes"
stamp "bound() fix"
NVARCH=$CC bash $H/boundfix.sh $NV $SRC $L/lib $W/bfx || fail boundfix
cp $W/bfx/libspumaBoundFix.so $L/libspumaBoundFix_spdp.so || fail "bound() fix copy"
tar cf - -C $L bin lib libspumaBoundFix_spdp.so | zstd -3 -T0 > $W/spumaS_sm$CC.tar.zst || fail pack
stamp "packed $(du -h $W/spumaS_sm$CC.tar.zst | cut -f1)"
install $W/spumaS_sm$CC.tar.zst
