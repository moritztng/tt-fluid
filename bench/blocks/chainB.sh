#!/bin/bash
# flx-kernel chainB (anx-glx, chip 2 = node 2): the block A/B on one chip. DECISIONS D26: chip 2 under
# flock ~/flx/locks/c2; flx-mg's hood arms have it 02:45-03:15Z, then this row.
#   A = stock op words (--blocks 0), B = generated bodies (--blocks 1, C_BLK), X = broken arm (--blocks 2).
# Part 1, pitzDaily (one chip, 50 its, legs of about a minute): from 02:26Z, lock per leg, no leg starts after PZEND.
# Part 2, hood k1 (hood65 on one chip, T 6, 20 its; --cores is a CPU-only proxy): not before HSTART. Then one --prof leg per arm (D22's rule).
# Series per case: B0 A0 (each image's first run: discarded), A1 B1 A2 B2, X. If arm A's first run fails with
# --levels 1, the series runs again on the images without --levels (pz0*, h0*).
# Per leg: no ~/fleet/HANDBACK, fresh HOLDER.windows mirror and no window within 9 min, HOLDER gives chip 2 to an flx
# row, node 2 closed, the lock for the leg only, timeout on the run, sysfs AICLK sampled every 2 s. Status: logs/chainB.status
K=$HOME/flx/kernel; T=$K/tree; R=$K/run; L=$K/logs; ST=$L/chainB.status
CHIPS=2; LOCK=$HOME/flx/locks/c2; SHA=$(cat $K/tree.sha)
. $R/ablib.sh
PZEND=$(date -u -d ${PZEND:-2026-11-28T02:43:00Z} +%s); HSTART=$(date -u -d ${HSTART:-2026-11-28T03:15:00Z} +%s)
GIVEUP=$(date -u -d ${GIVEUP:-2026-11-28T05:30:00Z} +%s)
FL="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --tsort 1 --l1 1"
FLP="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --l1 1"   # pitzDaily on a full chip is T 1, all in L1: no page streaming
XFH="--gcache 14 --bgts 5 --tgts 2 --sysfs-nodes $CHIPS"; XFP="--sysfs-nodes $CHIPS"
PZ="case 50 $FLP --levels 1 --case $T/cases/pitzDaily"; PZ0="case 50 $FLP --case $T/cases/pitzDaily"
HD="case 20 $FL --levels 1 --page 128 --case $HOME/flx/asm/hood65"; HD0="case 20 $FL --page 32 --case $HOME/flx/asm/hood65"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CHIPS
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto\")" 2>/dev/null | tail -n 1)
mkdir -p $HOME/flx/locks; exec 9>>$LOCK
st "chainB armed pid $$ chip $CHIPS tree $SHA pz until $(date -u -d @$PZEND +%T)Z hood from $(date -u -d @$HSTART +%T)Z gives up $(date -u -d @$GIVEUP +%T)Z desc $TT_MESH_GRAPH_DESC_PATH"
[ -e "$TT_MESH_GRAPH_DESC_PATH" ] || { st "no p150 mesh graph descriptor: not started"; exit 2; }

leg() { # NAME IMG "LOWERING FLAGS" BLOCKS LEGMAX END(epoch, 0 = none) EXTRA -> run_mesh's rc; 6 refused, 7 no lock, 9 past END
  local n=$1 rc a d ms w=0
  while :; do
    [ -e $HOME/fleet/HANDBACK ] && { st "$n refused: HANDBACK"; exit 3; }
    [ $(now) -gt $GIVEUP ] && { st "$n: gave up"; exit 5; }
    [ $6 != 0 ] && [ $(( $(now) + $5 )) -gt $6 ] && { st "$n not started: it would not end by $(date -u -d @$6 +%T)Z"; return 9; }
    [ -e $R/img/$2.ready ] || { [ $w = 0 ] && st "$n waits for image $2"; w=1; sleep 10; continue; }
    if win $n && holder $n && flock -w 60 9; then
      closed $n 30 && break
      flock -u 9
    fi
    sleep 20
  done
  sampler $L/$n.aiclk
  ( cd $T && PYTHONPATH=$T timeout $5 $PY -u -m ttfluid.device.run_mesh $3 --blocks $4 --from-images $R/img/$2 --noemu --runs 1 --os 1 $XF $7 --out $R/$n.jsonl > $L/$n.log 2>&1 9>&- ); rc=$?
  kill $SAMP 2>/dev/null
  flock -u 9
  a=$(aiclk $L/$n.aiclk)
  ms=$(grep -a -o "\"ms_device\": \[[0-9.]*" $R/$n.jsonl 2>/dev/null | tail -n 1 | cut -d"[" -f2)
  d=$(dig $(ls $R/${n}_case_*state.npz 2>/dev/null | head -n 1) 2>/dev/null)
  grep -a "^phase " $L/$n.log > $L/$n.phase 2>/dev/null
  st "$n rc=$rc img=$2 blocks=$4 host=anx-glx chip=$CHIPS sha=$SHA aiclk=$a ms_device=$ms digest=$d | $(grep -a "Error\|error:\|FATAL\|Traceback\|abort" $L/$n.log | head -n 2 | cut -c1-200 | tr "\n" " ")"
  return $rc
}
series() { # PREFIX "LOWERING FLAGS" LEGMAX END -> 0 done, 1 arm A failed, 2 arm B failed, 9 out of time
  local p=$1 r
  leg ${p}_A0 ${p}A "$2" 0 $3 $4; r=$?; [ $r = 9 ] && return 9; [ $r != 0 ] && return 1
  leg ${p}_B0 ${p}B "$2" 1 $3 $4; r=$?; [ $r = 9 ] && return 9; [ $r != 0 ] && return 2
  for k in 1 2; do
    leg ${p}_A$k ${p}A "$2" 0 $3 $4; [ $? = 9 ] && return 9
    leg ${p}_B$k ${p}B "$2" 1 $3 $4; [ $? = 9 ] && return 9
  done
  leg ${p}_X ${p}X "$2" 2 $3 $4; [ $? = 9 ] && return 9
  return 0
}
XF=$XFP
if [ -z "$SKIPPZ" ]; then
  series pz "$PZ" 150 $PZEND; r=$?; st "series pz -> $r"
  [ $r = 1 ] && { series pz0 "$PZ0" 150 $PZEND; st "series pz0 -> $?"; }
fi
while [ $(now) -lt $HSTART ]; do sleep 5; done
XF=$XFH
series h "$HD" 420 0; r=$?; st "series h -> $r"; H=h; HF=$HD
[ $r = 1 ] && { series h0 "$HD0" 420 0; r=$?; st "series h0 -> $r"; H=h0; HF=$HD0; }
if [ $r = 0 ]; then
  leg ${H}_Ap ${H}A "$HF" 0 600 0 --prof; leg ${H}_Bp ${H}B "$HF" 1 600 0 --prof
fi
st "chainB done"
