#!/bin/bash
# Box-side plan: k independent SPUMA SPDP cases at once on ONE GPU, HPC_motorbike Small (8,662,309 cells), forceCoeffs
# on, from the potentialFoam start, each case through spuma_run.sh exactly as the one-case-per-GPU baseline ran.
# A cell is <k>:<mode>[:<variant>[:<its>[:<need-s>]]]. Mode "mps" shares the GPU through CUDA MPS (kernels of different
# cases overlap), "off" leaves the default time-slicing. A variant runs fvSolution.spuma.mbs.<variant> instead of the
# baseline dictionary (tag P_<mode>_k<k>_<variant>; empty or "-" = baseline). <its> other than IT makes a timing screen
# (tag suffix _it<its>): compare its ClockTime with a full cell's at the same iteration, it carries no Cd grade.
# <need-s> is the time the cell must have left to start (default NEED, else 400 + 500 k); given in the cell, it also
# ends the cell at 1.5x that, so a screen that thrashes cannot eat the cells after it. "auto" picks the next k from the two MPS cells before it: 8 (or 6) while packing still
# pays and host RAM allows, else 3. The batch makespan (launch of the first case to exit of the last) is the number:
# cases per GPU-hour = k x 3600 / makespan.
# Usage: pack_plan.sh <work-dir> <deadline-seconds>    env CELLS ("1:off 2:mps 4:mps auto 2:off"), IT (800), GPUS ("0"), WRITE, CPG
# GPUS="0 1 ... 7" fills a whole box: a cell then runs k cases on EACH listed GPU in one batch (cases _c0.. in GPU
# order), so the makespan carries the shared host; per_case stays makespan / k, the per-GPU figure the price uses. One
# MPS control daemon serves every GPU (NVIDIA: one per node). WRITE=1 has every case write its final fields (binary)
# inside the wall, the counterpart of the state download a Tenstorrent run is charged.
# CPG=<n> pins the cases of each GPU to n physical cores of their own, one thread each, taken from the cores that are
# idlest when the plan starts: the host share the priced box has per GPU (2x 56 cores under 8 GPUs = 14), so a rented
# host with more cores cannot flatter a packed cell. Without it the cases float over every CPU the container may use.
# Expects in <work-dir>: mbs_np1/ (the case), libspumaBoundFix_spdp.so, and the spumaS bundle unpacked under /opt.
set -u
W=$1; DL=$2; T0=$(date +%s); H=$(cd $(dirname $0) && pwd)
IT=${IT:-800}; GS=${GPUS:-${GPU:-0}}; NG=$(echo $GS | wc -w); CELLS=${CELLS:-1:off 2:mps 4:mps auto 2:off}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
left() { echo $(( DL - ($(date +%s) - T0) - 60 )); }
ms() { awk -v t=$1 '$1=="CELL" && $2==t {print $3}' $W/cells.tsv | tail -1; }   # makespan of a finished cell, "" if none
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,clocks.max.mem,power.limit,driver_version,compute_mode --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # anything listed here is a co-tenant
lscpu | grep -E "Model name|^CPU\(s\)|Socket"; nproc; free -g | head -2
: > $W/cells.tsv
PIN=()   # PIN[j] = taskset list of the j-th GPU in GPUS
if [ -n "${CPG:-}" ]; then
  mapfile -t PIN < <(python3 $H/pin_cores.py $CPG $NG) && [ ${#PIN[@]} = $NG ] || { stamp "PLAN-FAILED: no $CPG idle physical cores for each of $NG GPUs"; exit 1; }
  stamp "CPG=$CPG: ${PIN[*]}"
fi

cell() {
  local k=$1 mode=$2 var=${3:-} it=${4:-$IT} need=${5:-${NEED:-$(( 400 + 500 * $1 ))}} tag fvs L pool g i j=0 pids=() ta tb
  [ "$var" = - ] && var=
  tag=P_${mode}_k$k${var:+_$var}; fvs=$H/fvSolution.spuma.mbs${var:+.$var}; [ $it != $IT ] && tag=${tag}_it$it
  L=$(left); [ $L -lt $need ] && { stamp "$tag skipped, ${L}s left, needs $need"; return; }
  [ -n "${5:-}" ] && [ $L -gt $(( need * 3 / 2 )) ] && L=$(( need * 3 / 2 ))
  # GiB of managed memory per case. Only touched pages are backed (a case touches ~6.1 GiB, 60 GiB pool or 9), so
  # past k=9 the pools oversubscribe the card on paper and the touched total (k x 6.1) is what has to fit.
  pool=$(( 72 / k )); [ $pool -gt 60 ] && pool=60; [ $pool -lt 8 ] && pool=8
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  if [ $mode = mps ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d; stamp "$tag MPS daemon rc=$?"
  fi
  stamp "$tag: $k cases x $it it on GPU $GS, pool $pool GiB each, ${L}s left"
  local gi=-1
  for g in $GS; do gi=$((gi+1)); for i in $(seq 1 $k); do
    R=$W/${tag}_c$j; rm -rf $R; mkdir -p $R; cp -al $W/mbs_np1/constant $W/mbs_np1/0 $R/; cp -r $W/mbs_np1/system $R/
    CUDA_VISIBLE_DEVICES=$g SPUMA=spumaS BFX=$W/libspumaBoundFix_spdp.so ${PIN[$gi]:+taskset -c ${PIN[$gi]}} timeout -s TERM $L \
      bash $H/spuma_run.sh $R $it $pool $fvs 1 ${WRITE:+$it} > $W/run_${tag}_c$j.out 2>&1 &
    pids+=($!); j=$((j+1))
  done; done
  ta=$(date +%s.%N)
  echo 0 > $W/rss_$tag.txt
  nvidia-smi -i ${GS// /,} --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader \
    -lms 1000 > $W/gpu_$tag.csv &
  local smi=$!
  # host RAM high-water of all solver processes, and one mid-run look at who is on the GPU (MPS clients show as M+C)
  ( m=0; while sleep 5; do r=$(ps -C simpleFoam -o rss= | awk '{s+=$1} END{print int(s/1024)}'); [ ${r:-0} -gt $m ] && m=$r && echo $m > $W/rss_$tag.txt; done ) &
  local rss=$!
  ( sleep 180; { nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
      pgrep -a nvidia-cuda-mps; [ $mode = mps ] && echo get_server_list | nvidia-cuda-mps-control; free -g | head -2; uptime; } > $W/smi_$tag.txt 2>&1 ) &
  local look=$!
  wait "${pids[@]}"; tb=$(date +%s.%N)
  kill $smi $rss $look 2>/dev/null
  [ $mode = mps ] && { echo quit | nvidia-cuda-mps-control; stamp "$tag MPS stopped"; }
  python3 - $W $tag $k $ta $tb $it $NG <<'EOF' | tee -a $W/cells.tsv
import glob, re, statistics as st, sys
W, tag, k, ta, tb, IT, NG = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
its, clock = [], []
for c in sorted(glob.glob(f"{W}/{tag}_c*"), key=lambda c: int(c.rsplit("_c", 1)[1])):
    log = open(f"{c}/log.simpleFoam", errors="replace").read()
    its.append(len(re.findall(r"^Time = ", log, re.M)))
    m = re.findall(r"ClockTime = (\d+) s", log); clock.append(int(m[-1]) if m else 0)
rows = [l.split(", ") for l in open(f"{W}/gpu_{tag}.csv") if l.count(",") == 5]
num = lambda s: float(s.split()[0])
p, sm, u, mem = ([num(r[i]) for r in rows] for i in (1, 2, 4, 5))
busy = [x for x, y in zip(p, u) if y >= 50] or [0]
srt = sorted(sm)
print("CELL", tag, f"{tb-ta:.1f}", f"per_case={(tb-ta)/k:.1f}", f"ok={sum(i == IT for i in its)}/{k * NG}", f"gpus={NG}", "its=" + ",".join(map(str, its)),
      "clock=" + ",".join(map(str, clock)), f"power_mean={st.mean(p):.1f}W", f"power_busy={st.mean(busy):.1f}W",
      f"util_mean={st.mean(u):.1f}%", f"sm_median={srt[len(srt)//2]:.0f}MHz", f"sm_at_max={100*sum(x >= max(sm) for x in sm)/len(sm):.0f}%",
      f"gpu_mem_max={max(mem):.0f}MiB", f"host_rss_max={open(f'{W}/rss_{tag}.txt').read().strip()}MiB")
EOF
  cat $W/${tag}_c*/summary.txt
  for R in $W/${tag}_c[0-9]*; do rm -rf $R/[1-9]* $R/constant $R/0; done   # keep logs and postProcessing
}

for c in $CELLS; do
  # a baseline-dictionary batch in which no case finished is a broken box or build, not a packing number: stop before
  # paying for more. A variant that finishes no case is that variant's result and the plan goes on.
  bad=$(grep -E "^CELL P_[a-z]+_k[0-9]+ .* ok=0/" $W/cells.tsv | cut -d' ' -f2)
  [ -n "$bad" ] && { stamp "PLAN-FAILED: $bad finished no case"; exit 1; }
  if [ $c = auto ]; then
    m2=$(ms P_mps_k2); m4=$(ms P_mps_k4); k=3
    if [ -n "$m2" ] && [ -n "$m4" ]; then
      memk=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo); r4=$(cat $W/rss_P_mps_k4.txt)
      # packing still pays from 2 to 4 (>= 5 % per case): go on to 8, or 6, as far as host RAM and the clock allow
      k=$(python3 -c "
m2, m4, mem, r4, left = $m2, $m4, $memk, $r4, $(left)
k = 3
if m4 / 4 < 0.95 * m2 / 2:
    for c in (8, 6):
        if c * r4 / 4 < 0.85 * mem and m4 * c / 4 * 1.15 + 200 < left: k = c; break
print(k)")
      NEED=$(python3 -c "print(int($m4 * $k / 4 * 1.15 + 200))")
    fi
    stamp "auto -> k=$k (mps k2 ${m2:-none} s, k4 ${m4:-none} s)"
    cell $k mps; unset NEED
  else
    IFS=: read -r ck cm cv ci cn <<< "$c"; cell $ck $cm "$cv" "$ci" "$cn"
  fi
done
stamp "PLAN-END"
