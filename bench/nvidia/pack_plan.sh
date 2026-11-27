#!/bin/bash
# Box-side plan: k independent SPUMA SPDP cases at once on ONE GPU, HPC_motorbike Small (8,662,309 cells), forceCoeffs
# on, from the potentialFoam start, each case through spuma_run.sh exactly as the one-case-per-GPU baseline ran.
# A cell is <k>:<mode>; mode "mps" shares the GPU through CUDA MPS (kernels of different cases overlap), "off" leaves
# the default time-slicing. "auto" picks the next k from the two MPS cells before it: 8 (or 6) while packing still
# pays and host RAM allows, else 3. The batch makespan (launch of the first case to exit of the last) is the number:
# cases per GPU-hour = k x 3600 / makespan.
# Usage: pack_plan.sh <work-dir> <deadline-seconds>    env CELLS ("1:off 2:mps 4:mps auto 2:off"), IT (800), GPU (0)
# Expects in <work-dir>: mbs_np1/ (the case), libspumaBoundFix_spdp.so, and the spumaS bundle unpacked under /opt.
set -u
W=$1; DL=$2; T0=$(date +%s); H=$(cd $(dirname $0) && pwd)
IT=${IT:-800}; G=${GPU:-0}; CELLS=${CELLS:-1:off 2:mps 4:mps auto 2:off}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
left() { echo $(( DL - ($(date +%s) - T0) - 60 )); }
ms() { awk -v t=$1 '$1=="CELL" && $2==t {print $3}' $W/cells.tsv | tail -1; }   # makespan of a finished cell, "" if none
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,clocks.max.mem,power.limit,driver_version,compute_mode --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # anything listed here is a co-tenant
lscpu | grep -E "Model name|^CPU\(s\)|Socket"; nproc; free -g | head -2
: > $W/cells.tsv

cell() {
  local k=$1 mode=$2 tag=P_${2}_k$1 L pool i pids=() ta tb
  L=$(left); [ $L -lt $(( ${NEED:-$(( 400 + 500 * k ))} )) ] && { stamp "$tag skipped, ${L}s left"; return; }
  pool=$(( 72 / k )); [ $pool -gt 60 ] && pool=60   # GiB of managed memory per case; a case touches ~6.2 GiB
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  if [ $mode = mps ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d; stamp "$tag MPS daemon rc=$?"
  fi
  stamp "$tag: $k cases x $IT it, pool $pool GiB each, ${L}s left"
  for i in $(seq 0 $((k-1))); do
    R=$W/${tag}_c$i; rm -rf $R; mkdir -p $R; cp -al $W/mbs_np1/constant $W/mbs_np1/0 $R/; cp -r $W/mbs_np1/system $R/
    CUDA_VISIBLE_DEVICES=$G SPUMA=spumaS BFX=$W/libspumaBoundFix_spdp.so timeout -s TERM $L \
      bash $H/spuma_run.sh $R $IT $pool $H/fvSolution.spuma.mbs 1 > $W/run_${tag}_c$i.out 2>&1 &
    pids+=($!)
  done
  ta=$(date +%s.%N)
  echo 0 > $W/rss_$tag.txt
  nvidia-smi -i $G --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader \
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
  python3 - $W $tag $k $ta $tb $IT <<'EOF' | tee -a $W/cells.tsv
import glob, re, statistics as st, sys
W, tag, k, ta, tb, IT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
its, clock = [], []
for c in sorted(glob.glob(f"{W}/{tag}_c*")):
    log = open(f"{c}/log.simpleFoam", errors="replace").read()
    its.append(len(re.findall(r"^Time = ", log, re.M)))
    m = re.findall(r"ClockTime = (\d+) s", log); clock.append(int(m[-1]) if m else 0)
rows = [l.split(", ") for l in open(f"{W}/gpu_{tag}.csv") if l.count(",") == 5]
num = lambda s: float(s.split()[0])
p, sm, u, mem = ([num(r[i]) for r in rows] for i in (1, 2, 4, 5))
busy = [x for x, y in zip(p, u) if y >= 50] or [0]
srt = sorted(sm)
print("CELL", tag, f"{tb-ta:.1f}", f"per_case={(tb-ta)/k:.1f}", f"ok={sum(i == IT for i in its)}/{k}", "its=" + ",".join(map(str, its)),
      "clock=" + ",".join(map(str, clock)), f"power_mean={st.mean(p):.1f}W", f"power_busy={st.mean(busy):.1f}W",
      f"util_mean={st.mean(u):.1f}%", f"sm_median={srt[len(srt)//2]:.0f}MHz", f"sm_at_max={100*sum(x >= max(sm) for x in sm)/len(sm):.0f}%",
      f"gpu_mem_max={max(mem):.0f}MiB", f"host_rss_max={open(f'{W}/rss_{tag}.txt').read().strip()}MiB")
EOF
  cat $W/${tag}_c*/summary.txt
  for i in $(seq 0 $((k-1))); do rm -rf $W/${tag}_c$i/[1-9]* $W/${tag}_c$i/constant $W/${tag}_c$i/0; done   # keep logs and postProcessing
}

for c in $CELLS; do
  # a batch in which no case finished is a broken box or build, not a packing number: stop before paying for more
  grep -q "^CELL .* ok=0/" $W/cells.tsv && { stamp "PLAN-FAILED: $(grep '^CELL .* ok=0/' $W/cells.tsv | cut -d' ' -f2) finished no case"; exit 1; }
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
    cell ${c%:*} ${c#*:}
  fi
done
stamp "PLAN-END"
