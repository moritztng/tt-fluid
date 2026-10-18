#!/bin/bash
# Stop SPUMA's full Allwmake and build only what simpleFoam (+ forces/field function objects) needs.
# Usage: spuma_min.sh [stop]   SPUMA_DIR/SPUMA_CC select the tree (default /opt/spuma, 80).
D=${SPUMA_DIR:-/opt/spuma}
if [ "${1:-}" = stop ]; then
  for p in $(pgrep -f "spuma.sh|Allwmake|wmake|make -j|make --no-print"); do [ $p != $$ ] && kill -INT $p 2>/dev/null; done
  sleep 5; pkill -INT -f nvc++ ; pkill -INT -f nvcpfe; sleep 3
fi
export PATH=$(ls -d ${NVROOT:-/opt/nvidia/hpc_sdk}/Linux_x86_64/2*/compilers/bin | sort -V | tail -1):$PATH
export NVARCH=${SPUMA_CC:-80} have_cuda=true
set +eu; source $D/etc/bashrc; set -eu
cd $D/src
t0=$(date +%s)
for t in TurbulenceModels/schemes overset fvOptions atmosphericModels conversion functionObjects/field functionObjects/forces functionObjects/utilities functionObjects/initialisation; do
  echo "== $t +$(( $(date +%s) - t0 ))s"; wmake -j ${NP:-46} -s -q libso $t 2>&1 | grep -E "Error|error:" | head -5
done
cd $D/applications
for t in solvers/incompressible/simpleFoam solvers/basic/potentialFoam; do
  echo "== $t +$(( $(date +%s) - t0 ))s"; wmake -j ${NP:-46} -s -q $t 2>&1 | grep -E "Error|error:" | head -5
done
ls -la $FOAM_APPBIN; echo "done +$(( $(date +%s) - t0 ))s"
