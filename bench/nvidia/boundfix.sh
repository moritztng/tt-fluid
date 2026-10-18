#!/bin/bash
# Build libspumaBoundFix.so (boundfix/) against the SPUMA source tree, without a GPU.
# Usage: boundfix.sh <nvhpc-root> <spuma-src> <spuma-runtime-libbin> <out-dir>
# The runtime libbin is opt/spumaM/platforms/linux64NvidiaDPInt32Opt/lib from spumaM_rt.tar.zst.
set -euo pipefail
NV=$1 SRC=$2 RT=$3 OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$(ls -d $NV/Linux_x86_64/2*/compilers/bin | sort -V | tail -1):$PATH
export have_cuda=true NVARCH=${NVARCH:-80}
sed -i -E 's/^(export WM_COMPILER=).*/\1Nvidia/' $SRC/etc/bashrc
set +eu; set --; source $SRC/etc/bashrc; set -eu
[ -x $WM_PROJECT_DIR/wmake/platforms/$WM_ARCH$WM_COMPILER/wmkdepend ] || (cd $WM_PROJECT_DIR/wmake/src && make -s)
for d in OpenFOAM OSspecific/POSIX finiteVolume meshTools; do wmakeLnInclude -u $SRC/src/$d >/dev/null; done
rm -rf $OUT/build && mkdir -p $OUT && cp -r $HERE/boundfix $OUT/build
# exports.map names the DP symbol; scalar is float under SP and SPDP
[ "$WM_PRECISION_OPTION" = DP ] || sed -i 's/<double/<float/g' $OUT/build/exports.map
export FOAM_USER_LIBBIN=$OUT SPUMA_RT_LIBBIN=$RT
cd $OUT/build && wmake libso 2>&1 | tail -20
ls -la $OUT/libspumaBoundFix.so
nm -DC $OUT/libspumaBoundFix.so | grep ' T Foam::bound('
