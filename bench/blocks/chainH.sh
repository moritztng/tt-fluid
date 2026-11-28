#!/bin/bash
# flx-kernel chainH (anx-glx, chip 2 = node 2): the hood k1 block A/B (hood65 on one chip, T 6, 20 its) around the
# whole-box window 03:45-04:30Z (DECISIONS D31: no FLX node open from 03:42Z; D30: a leg is bounded by its iterations,
# the timeout is only the net under a hang).
#   A = stock op words (--blocks 0), B = generated bodies (--blocks 1, C_BLK), X = broken arm (--blocks 2).
# Legs in order: A0 B0 (each image's first run: discarded) A1 B1, the --prof legs Ap Bp (D22's rule), X, A2 B2.
# A leg starts only if the last leg's wall + 30 s (300 s before the first) ends by E1. What is left runs after the
# window, behind one cold pair A0c B0c (the first runs after the window's reset: discarded).
# If arm A's first run fails with --levels 1 the list restarts on the images without it (h0*). If arm B fails the chain ends.
# Per leg: no ~/fleet/HANDBACK, fresh HOLDER.windows mirror and no window within 9 min, HOLDER gives chip 2 to an flx
# row, node 2 closed, flock ~/flx/locks/c2 for the leg only, sysfs AICLK sampled every 2 s. Status: logs/chainB.status
K=$HOME/flx/kernel; T=$K/tree; R=$K/run; L=$K/logs; ST=$L/chainB.status
CHIPS=2; LOCK=$HOME/flx/locks/c2; SHA=$(cat $K/tree.sha)
. $R/ablib.sh
HSTART=$(date -u -d ${HSTART:-2026-11-28T03:15:00Z} +%s); E1=$(date -u -d ${E1:-2026-11-28T03:41:30Z} +%s)
GIVEUP=$(date -u -d ${GIVEUP:-2026-11-28T06:00:00Z} +%s)
FL="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --tsort 1 --l1 1"
XF="--gcache 14 --bgts 5 --tgts 2 --sysfs-nodes $CHIPS"
HD="case 20 $FL --levels 1 --page 128 --case $HOME/flx/asm/hood65"; HD0="case 20 $FL --page 32 --case $HOME/flx/asm/hood65"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CHIPS
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto\")" 2>/dev/null | tail -n 1)
mkdir -p $HOME/flx/locks; exec 9>>$LOCK
st "chainH armed pid $$ chip $CHIPS tree $SHA hood from $(date -u -d @$HSTART +%T)Z, off the node by $(date -u -d @$E1 +%T)Z, the rest after the window; gives up $(date -u -d @$GIVEUP +%T)Z"
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
LEGS=("A0 A 0 420" "B0 B 1 420" "A1 A 0 420" "B1 B 1 420" "Ap A 0 600 --prof" "Bp B 1 600 --prof" "X X 2 420" "A2 A 0 420" "B2 B 1 420")
P=h; F=$HD; I=0
list() { # END -> 0 the list is through, 9 out of time before END; exits when an arm is broken
  local r
  while [ $I -lt ${#LEGS[@]} ]; do
    set -- ${LEGS[$I]}
    leg ${P}_$1 ${P}$2 "$F" $3 $4 $E "$5"; r=$?
    [ $r = 9 ] && return 9
    if [ $I = 0 ] && [ $r != 0 ]; then
      [ $P = h0 ] && { st "arm A failed on h0 as well (rc $r): chain ends"; exit 6; }
      st "arm A's first run failed on the level-fused images (rc $r): the list restarts on h0"; P=h0; F=$HD0; continue
    fi
    [ $I = 1 ] && [ $r != 0 ] && { st "arm B's first run failed (rc $r): chain ends, logs/${P}_B0.log"; exit 7; }
    [ $r = 8 ] && { st "image missing for ${P}_$1: leg skipped"; }
    I=$((I + 1))
  done
  return 0
}
while [ $(now) -lt $HSTART ]; do sleep 5; done
E=$E1; list; r=$?; st "before the window: $I of ${#LEGS[@]} legs (series $P) -> $r"
if [ $r = 9 ]; then
  while [ $(now) -lt $E1 ] || ! winq; do
    [ -e $HOME/fleet/HANDBACK ] && { st "refused: HANDBACK"; exit 3; }
    [ $(now) -gt $GIVEUP ] && { st "gave up waiting for the window's end"; exit 5; }
    sleep 20
  done
  st "window over: one cold pair, then the remaining legs"
  E=0; EST=300
  leg ${P}_A0c ${P}A "$F" 0 420 0 ""; leg ${P}_B0c ${P}B "$F" 1 420 0 ""
  list; st "after the window: $I of ${#LEGS[@]} legs -> $?"
fi
st "chainH done"
