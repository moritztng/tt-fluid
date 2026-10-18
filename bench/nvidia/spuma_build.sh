#!/bin/bash
# Minimal SPUMA build (libraries up to TurbulenceModels + spuma_min.sh) as one fat binary for several GPU archs.
# Usage: spuma_build.sh <tree> "<cc list, e.g. 80 90 100>"   (no GPU needed; NVROOT = NVHPC root, LOG = log dir, NP = jobs)
set -u
H=$(cd $(dirname $0) && pwd); D=$1; CCS=${2:-"80 90"}
t0=$(date +%s); stamp(){ echo "[build +$(( $(date +%s) - t0 ))s] $*"; }
first=${CCS%% *}
# nvc++ takes -gpu=cc80,cc90; the nvcc rule gets one -gencode per arch.
GEN=$(for c in $CCS; do printf -- '-gencode arch=compute_%s,code=sm_%s ' $c $c; done)
sed -i -E "s|^cuARCH +:=.*|cuARCH    :=  -m64 $GEN|" $D/wmake/rules/General/cuda
export NVARCH=$(echo $CCS | sed 's/ /,cc/g')   # "80,cc90" -> -gpu=cc80,cc90
sed -i -E 's/^(export FOAM_SIGFPE=).*/\1false/' $D/etc/bashrc
grep -q '^export FOAM_SIGFPE' $D/etc/bashrc || echo 'export FOAM_SIGFPE=false' >> $D/etc/bashrc
export PATH=$(ls -d ${NVROOT:-/opt/nvidia/hpc_sdk}/Linux_x86_64/2*/compilers/bin | sort -V | tail -1):$PATH have_cuda=true
set +eu; source $D/etc/bashrc; set -u
stamp "WM_PROJECT_DIR=$WM_PROJECT_DIR NVARCH=$NVARCH cuARCH=$GEN"
(cd $D/wmake/src && ./Allmake) > ${LOG:-/root}/spuma_build.wmake.log 2>&1; stamp "wmake tools rc=$?"
cd $D/src
awk '{print} /^TurbulenceModels\/Allwmake/{exit}' Allwmake > Allwmake.min && chmod +x Allwmake.min
./Allwmake.min -j ${NP:-46} -s -q > ${LOG:-/root}/spuma_build.libs.log 2>&1; stamp "libs rc=$?"
SPUMA_DIR=$D SPUMA_CC=$NVARCH bash $H/spuma_min.sh; stamp done
