#!/bin/bash
# Box-side plan, vrt-fluid occDrivAer mid (5.56 M) SPUMA leg. SPDP (spumaS, the motorbike's best GPU path) for the
# graded 4000-it case with forceCoeffs on, then fp64 (spumaM) for 800 its as the precision control.
# Usage: midplan.sh <case-dir> <iters> <pool-GiB> <deadline-seconds>
# Bundles under /opt (spumaM, spumaS) and both fix libs in this dir, unpacked by session_mid.sh.
set -u
C=$1; IT=$2; POOL=$3; DL=$4; H=$(cd $(dirname $0) && pwd)
cp -a $C ${C}_fp64
SPUMA=spumaS BFX=$H/libspumaBoundFix_S.so timeout $((DL * 55 / 100)) bash $H/spuma_run.sh $C $IT $POOL $H/fvSolution.spuma.occ 1 > $H/run_spdp.out 2>&1
cat $H/run_spdp.out
SPUMA=spumaM BFX=$H/libspumaBoundFix.so timeout $((DL * 40 / 100)) bash $H/spuma_run.sh ${C}_fp64 800 $POOL $H/fvSolution.spuma.occ 1 > $H/run_fp64.out 2>&1
cat $H/run_fp64.out
