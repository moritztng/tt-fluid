#!/bin/bash
# SPUMA simpleFoam on one GPU, or one rank per GPU, on a case initialised on the CPU.
# Usage: spuma_run.sh <case-dir> <iters> <pool-GiB|dummy> [fvSolution-file] [function-objects 0|1] [writeInterval]
# pool "dummy" = SPUMA's dummyMemoryPool (one cudaMallocManaged per allocation). The run starts from the latest time.
# Expects a SPUMA runtime bundle under /opt (SPUMA env: spumaM = fp64, spumaS = SPDP mixed; plus /opt/nvrt) and its
# bound() fix lib (BFX env, default libspumaBoundFix.so beside this script).
# Function objects are off by default (-noFunctionObjects); with 1 the case's own set runs, as in the CPU reference,
# and postProcessing/ carries the Cd history. Only forceCoeffsAll is kept (the case's coded wallClockTimeStatistics
# needs a compiler; probes/minMax/wallShearStress are extra work the GPU is spared, which flatters NVIDIA, not us). A NaN in the log stops the solver at once (SIGTERM).
# NP (env, default 1) > 1 runs a decomposed case with one MPI rank per GPU (MPIRUN names the launcher).
# WRAP (env) prefixes simpleFoam with a profiler command. Writes log.simpleFoam, gpu.csv (1 s nvidia-smi samples), iters.tsv and summary.txt in the case.
set -u
C=$1; IT=$2; POOL=$3; FVS=${4:-}; FO=${5:-0}; WI=${6:-100000}
BF=${BFX:-$(cd $(dirname $0) && pwd)/libspumaBoundFix.so}  # boundfix/: SPUMA's device min() misses negative cells
S=/opt/${SPUMA:-spumaM}
cd $C
[ -n "$FVS" ] && cp $FVS system/fvSolution
sed -i -E "s/^endTime .*/endTime $IT;/; s/^writeInterval .*/writeInterval $WI;/" system/controlDict
P=$(ls -d $S/platforms/linux64Nvidia*Int32Opt | head -1)
set --  # OpenFOAM's bashrc reads positional arguments as config overrides
set +u; export WM_COMPILER=Nvidia; source $S/etc/bashrc >/dev/null 2>&1; set -u
# NVHPC 26.9 links CUDA 13.3; an older host driver takes the bundled forward-compat libcuda, a newer one must not
# (compat 610.43 under host 610.57: SIGFPE before "Create time", 2026-10-12).
HV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); CV=$(ls /opt/nvrt/compat/libcuda.so.*.* | sed 's/.*libcuda.so.//')
COMPAT=/opt/nvrt/compat:; [ "$(printf '%s\n' $HV $CV | sort -V | head -1)" = "$CV" ] && COMPAT=
export PATH=$P/bin:$PATH LD_LIBRARY_PATH=$COMPAT/opt/nvrt:$P/lib/sys-openmpi:$P/lib:${LD_LIBRARY_PATH:-}
which simpleFoam; ldd $P/bin/simpleFoam | grep "not found"
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader \
  -lms 1000 > gpu.csv &
SMI=$!
NOFO=-noFunctionObjects
[ "$FO" = 1 ] && NOFO= && python3 - <<'PY'
import re
t = open("system/controlDict").read()
i = t.index("\nfunctions"); blk = re.search(r"\n    forceCoeffsAll\n    \{.*?\n    \}\n", t, re.S).group(0)
open("system/controlDict", "w").write(t[:i] + "\nfunctions\n{" + blk + "}\n")
PY
t0=$(date +%s.%N)
POOLOPT="-pool fixedSizeMemoryPool -poolSize $POOL"; [ "$POOL" = dummy ] && POOLOPT="-pool dummyMemoryPool"
PRE=""; [ -f $BF ] && PRE=$BF && echo "LD_PRELOAD $BF"
NP=${NP:-1}
if [ $NP -gt 1 ]; then
  # the fix lib goes to the ranks only (-x VAR=value), mpirun itself cannot load it
  ${MPIRUN:-mpirun} -np $NP --allow-run-as-root --bind-to none ${PRE:+-x LD_PRELOAD=$PRE} -x PATH -x LD_LIBRARY_PATH \
    simpleFoam -parallel $NOFO $POOLOPT > log.simpleFoam 2>&1 &
elif [ -n "${WRAP:-}" ]; then  # a profiler wrapper; the fix lib goes to simpleFoam only, not the profiler
  $WRAP ${PRE:+-e LD_PRELOAD=$PRE} simpleFoam $NOFO $POOLOPT > log.simpleFoam 2>&1 &
else
  LD_PRELOAD=$PRE simpleFoam $NOFO $POOLOPT > log.simpleFoam 2>&1 &
fi
SF=$!; trap 'kill -TERM $SF' TERM   # an outer timeout still gets the summary
while kill -0 $SF 2>/dev/null; do
  grep -qi "residual = -\?nan" log.simpleFoam && { echo "NaN at $(grep -c '^Time =' log.simpleFoam), stopping"; kill -TERM $SF; }
  sleep 10
done
wait $SF; echo "simpleFoam rc=$?"
t1=$(date +%s.%N)
kill $SMI
awk '/^Time = /{t=$3} /^ExecutionTime/{print t"\t"$3}' log.simpleFoam > iters.tsv
python3 - "$C" "$t0" "$t1" <<'EOF' | tee summary.txt
import sys, statistics as st
C, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = [tuple(map(float, l.split())) for l in open("iters.tsv")]
dt = [b[1] - a[1] for a, b in zip(rows, rows[1:])]
warm = dt[len(dt) // 5:] or [0]
print(f"{C} iters={len(rows)} wall={t1-t0:.1f}s first_it={rows[0][1] if rows else 0:.1f}s s/iter median={st.median(warm):.4f} "
      f"mean={st.mean(warm):.4f} min={min(warm):.4f} max={max(warm):.4f}")
EOF
