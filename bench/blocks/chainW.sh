#!/bin/bash
# flx-kernel chainW (anx-glx, chip 2 = node 2): who MATH waits for at its loads and acquires (run_mesh --prof 2,
# CFD_PROFW: census.mwait), on the hood k1 with the generated bodies (hood65 on one chip, T 6, 20 its, image hB).
# Legs: B0c (first run after the whole-box window's reset: discarded), Bw1 Bw2 (--prof 2), Bp (--prof, the
# instrument's own cost against it), B1 (plain). Every leg: ~/flx/winwait.sh, no ~/fleet/HANDBACK, HOLDER gives chip 2
# to an flx row, node 2 closed, flock ~/flx/locks/c2 for the leg only, sysfs AICLK sampled every 2 s. A leg is bounded
# by its 20 iterations; the timeout is only the net under a hang (D30). Gate: state digest 0b2e91c2aaf1689c on every leg.
# Status: logs/chainW.status
K=$HOME/flx/kernel; T=$K/tree4; R=$K/run; L=$K/logs; ST=$L/chainW.status
CHIPS=2; LOCK=$HOME/flx/locks/c2; SHA=$(cat $K/tree4.sha)
. $R/ablib.sh
HSTART=$(date -u -d ${HSTART:-2026-11-28T03:15:00Z} +%s); E1=$(date -u -d ${E1:-2026-11-28T03:41:30Z} +%s)
GIVEUP=$(date -u -d ${GIVEUP:-2026-11-28T06:00:00Z} +%s)
FL="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --tsort 1 --l1 1"
XF="--gcache 14 --bgts 5 --tgts 2 --sysfs-nodes $CHIPS"
HD="case 20 $FL --levels 1 --page 128 --case $HOME/flx/asm/hood65"; HD0="case 20 $FL --page 32 --case $HOME/flx/asm/hood65"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CHIPS
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto\")" 2>/dev/null | tail -n 1)
mkdir -p $HOME/flx/locks; exec 9>>$LOCK
st "chainW armed pid $$ chip $CHIPS tree $SHA: hood k1 with bodies (img hB), cold leg, two --prof 2 legs, one --prof leg, one plain leg, each behind winwait.sh; gives up $(date -u -d @$GIVEUP +%T)Z"
[ -e "$TT_MESH_GRAPH_DESC_PATH" ] || { st "no p150 mesh graph descriptor: not started"; exit 2; }
EST=300

imgfail() { # IMG -> 0 once the newest hprep has failed this image or has ended without it
  awk -v i="$1" '/hprep start/ {f = 0; d = 0} $2 == "img" && $3 == i && $4 != "rc=0" {f = 1} /hprep done/ {d = 1} END {exit !(f || d)}' $L/prep.status
}
winq() { # 0 = no booked window of another row now or within 9 min (ablib's win without its status line)
  local t=$(now) slug a b rest ta tb
  while read -r slug a b rest; do
    [ -z "$slug" ] && continue
    case "$slug" in \#*|flx-kernel) continue;; esac
    ta=$(date -u -d "$a" +%s 2>/dev/null); tb=$(date -u -d "$b" +%s 2>/dev/null)
    [ -n "$ta" ] && [ -n "$tb" ] && [ $((t + 540)) -ge "$ta" ] && [ "$t" -le "$tb" ] && return 1
  done < $HOME/fleet/HOLDER.windows
  return 0
}
leg() { # NAME IMG "LOWERING FLAGS" BLOCKS LEGMAX END(epoch, 0 = none) EXTRA -> run_mesh's rc; 8 image not built, 9 it would not end by END
  local n=$1 rc a d ms w=0 t0 tmo
  while :; do
    [ -e $HOME/fleet/HANDBACK ] && { st "$n refused: HANDBACK"; exit 3; }
    [ $(now) -gt $GIVEUP ] && { st "$n: gave up"; exit 5; }
    [ $6 != 0 ] && [ $(( $(now) + EST )) -gt $6 ] && { st "$n not started: $EST s would not end by $(date -u -d @$6 +%T)Z"; return 9; }
    [ -e $R/img/$2.ready ] || { imgfail $2 && { st "$n: image $2 was not built"; return 8; }; [ $w = 0 ] && st "$n waits for image $2"; w=1; sleep 10; continue; }
    if winq && win $n && holder $n && flock -w 60 9; then
      if closed $n 30; then
        [ $6 = 0 ] || [ $(( $(now) + EST )) -le $6 ] && break
      fi
      flock -u 9
    fi
    sleep 20
  done
  tmo=$5; [ $6 != 0 ] && [ $(( $6 - $(now) )) -lt $tmo ] && tmo=$(( $6 - $(now) ))
  t0=$(now); sampler $L/$n.aiclk
  ( cd $T && PYTHONPATH=$T timeout $tmo $PY -u -m ttfluid.device.run_mesh $3 --blocks $4 --from-images $R/img/$2 --noemu --runs 1 --os 1 $XF $7 --out $R/$n.jsonl > $L/$n.log 2>&1 9>&- ); rc=$?
  kill $SAMP 2>/dev/null
  flock -u 9
  EST=$(( $(now) - t0 + 30 ))
  a=$(aiclk $L/$n.aiclk)
  ms=$(grep -a -o "\"ms_device\": \[[0-9.]*" $R/$n.jsonl 2>/dev/null | tail -n 1 | cut -d"[" -f2)
  d=$(dig $(ls $R/${n}_case_*state.npz 2>/dev/null | head -n 1) 2>/dev/null)
  grep -a "^phase " $L/$n.log > $L/$n.phase 2>/dev/null
  st "$n rc=$rc img=$2 blocks=$4 host=anx-glx chip=$CHIPS sha=$SHA aiclk=$a ms_device=$ms digest=$d wall=$(( $(now) - t0 ))s | $(grep -a "Error\|error:\|FATAL\|Traceback\|abort" $L/$n.log | head -n 2 | cut -c1-200 | tr "\n" " ")"
  [ $rc = 124 ] && st "$n was ended by the timeout ($tmo s): chip $CHIPS needs an open check before its next user (D30)"
  return $rc
}
E=0; EST=420
run1() { # leg's arguments; LEGMAX is also winwait's lead
  $HOME/flx/winwait.sh $5 $CHIPS >> $L/chainW.winwait 2>&1; local r=$?
  [ $r != 0 ] && { st "$1: winwait rc=$r: chain ends"; exit 4; }
  leg "$@"
}
# LEGS (default all five), PFX (default w): a re-run of some legs keeps the first chain's files
PFX=${PFX:-w}
for l in ${LEGS:-B0c Bw1 Bw2 Bp B1}; do
  case $l in
    B0c) run1 ${PFX}_B0c hB "$HD" 1 420 0 "" || { st "the cold body leg failed: chain ends, logs/${PFX}_B0c.log"; exit 7; };;
    Bw*) run1 ${PFX}_$l hB "$HD" 1 900 0 "--prof 2" || { st "the --prof 2 leg $l failed: chain ends, logs/${PFX}_$l.log"; exit 8; };;
    Bp) run1 ${PFX}_Bp hB "$HD" 1 600 0 "--prof";;
    B1) run1 ${PFX}_B1 hB "$HD" 1 420 0 "";;
  esac
done
for n in ${PFX}_Bw1 ${PFX}_Bw2; do [ -e $L/$n.log ] && grep -a "^census mwait" $L/$n.log | tail -n 1 > $L/$n.mwait; done
st "chainW done"
