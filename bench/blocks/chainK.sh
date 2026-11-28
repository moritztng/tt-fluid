#!/bin/bash
# flx-kernel chainK (anx-glx, pair 6,7 = nodes 6,7): the Small k2 block A/B against flx-sol's kept control
# (anx ~/flx/sol/img/small_k2_jac, main c22145e, 160.94 s device per 100 its, state f95cbd4a993426ef). D26: every leg
# under flock ~/flx/locks/c67, legs of 100 its or less, A B A B, first leg discarded. D30: a leg is bounded by its
# iterations; the timeout is only the net under a hang, and a leg it ends stops the chain. D31: no FLX node open
# inside a booked window.
#   A = stock op words (--blocks 0), B = generated bodies (--blocks 1, C_BLK), X = broken arm (--blocks 2).
# Legs (env LEGS, NAME:ARM ...): A0 B0 (each image's first run: discarded) A1 B1 A2 B2, then X if its image exists.
# In front of every leg (D34.3): ~/flx/slotwait.sh kernel <pair> <leg s>, ~/flx/winwait.sh, then the pair's flock. flx-sol's and flx-asm's
# protocol to the letter (tools/leg.sh, exp7b.sh): same run flags, same start state, same 1x2 descriptor.
# A leg takes 215-231 s of process wall on the control; one starts only if the last leg's wall + 30 s (300 s before
# the first) ends by END and 3 minutes before the next booked window; a leg that does not fit before a window waits
# for the window to pass, one that does not fit before END ends the chain.
# env: START, END (UTC stamps; END unset = no bound), CHIPS, LOCK, ITS, LEGTMO, GIVEUP, DRY=1 (prints the commands,
#      opens nothing, takes no lock; status logs/chainKdry.status),
#      TREE  the tree BOTH arms run from (default run/sktree = 8503079, the inline C_BLK form; its sha in TREE.sha).
#            An image encoded by one form of C_BLK must run on that form's vm_compute.
#      AIMG  arm A's image (default run/img/skA when it is ready, else the control), AX its extra flags (`none`)
#      BIMG  arm B's image (default run/img/skB_inline), BX its extra flags (default "--page 32"; `none` for none)
#      XIMG  the broken arm's image (default run/img/skX_inline; skipped while it has no .ready), XX (default BX)
# Status: logs/chainK.status, one line a leg with host= chip= sha= aiclk= n= and the state digest;
# `python3 run/absum.py logs/chainK.status sk` prints the lines with the in-run clock, the means and the digest gate.
K=$HOME/flx/kernel; R=$K/run; L=$K/logs; SB=$HOME/flx/sol; ST=$L/chainK.status; [ -n "$DRY" ] && ST=$L/chainKdry.status
CHIPS=${CHIPS:-6,7}; PAIR=c${CHIPS//,/}; LOCK=${LOCK:-$HOME/flx/locks/$PAIR}; ITS=${ITS:-100}; LEGTMO=${LEGTMO:-480}
LEGS=${LEGS:-A0:A B0:B A1:A B1:B A2:A B2:B X:X}
T=${TREE:-$R/sktree}; SHA=$(cat $T.sha 2>/dev/null)
CTL=$SB/img/small_k2_jac; CDIG=f95cbd4a993426ef94b88ce31aacccbe458fa9eb4ab85e86dce8cf78b6ed10db
if [ -z "$AIMG" ]; then AIMG=$CTL; [ -e $R/img/skA.ready ] && AIMG=$R/img/skA; fi
AX=${AX:-none}; BIMG=${BIMG:-$R/img/skB_inline}; BX=${BX:---page 32}; XIMG=${XIMG:-$R/img/skX_inline}; XX=${XX:-$BX}
[ "$AX" = none ] && AX=""; [ "$BX" = none ] && BX=""; [ "$XX" = none ] && XX=""
. $R/ablib.sh
START=$(date -u -d ${START:-now} +%s); [ -n "$END" ] && END=$(date -u -d $END +%s) || END=0
GIVEUP=$(date -u -d ${GIVEUP:-2026-11-28T08:00:00Z} +%s)
FL="--arch bh --mesh 1,2 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --forces $SB/case/small_fc_controlDict:forceCoeffsAll --case $SB/case/small_tt"
RUN="--runs 1 --noemu --nwin 1024 --links 0 --rlink 1 --os 1 --tsort 1 --gcache 14 --bgts 5 --tgts 2 --l1 1 --sysfs-nodes $CHIPS"
export TT_VISIBLE_DEVICES=$CHIPS TT_MESH_GRAPH_DESC_PATH=$SB/tools/bh_glx_1x2.textproto
mkdir -p $HOME/flx/locks; [ -z "$DRY" ] && exec 9>>$LOCK
EST=300

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
nextwin() { # epoch of the nearest booked window of another row that is still ahead, 0 if none
  local t=$(now) slug a b rest ta m=0
  while read -r slug a b rest; do
    [ -z "$slug" ] && continue
    case "$slug" in \#*|flx-kernel) continue;; esac
    ta=$(date -u -d "$a" +%s 2>/dev/null); [ -z "$ta" ] && continue
    [ "$ta" -gt "$t" ] && { [ $m = 0 ] || [ "$ta" -lt $m ]; } && m=$ta
  done < $HOME/fleet/HOLDER.windows
  echo $m
}
bound() { # -> the epoch a leg must have ended by: END, or 3 minutes before the next booked window if that is earlier; 0 = none
  local e=$END w=$(nextwin)
  if [ $w != 0 ]; then w=$((w - 180)); { [ $e = 0 ] || [ $w -lt $e ]; } && e=$w; fi
  echo $e
}
leg() { # NAME ARM -> run_mesh's rc; 8 the arm's image is not ready, 9 it would not end by the bound
  local n=$1 arm=$2 img b x rc a d ms same tmo t0 S cmd e
  case $arm in A) img=$AIMG; b=0; x=$AX;; B) img=$BIMG; b=1; x=$BX;; X) img=$XIMG; b=2; x=$XX;; esac
  [ -e $img.ready ] || { st "$n skipped: $img.ready is missing"; return 8; }
  cmd="$PY -u -m ttfluid.device.run_mesh case $ITS $FL $RUN $x --blocks $b --from-images $img --state-in $SB/case/st_small_pf_k4_n.npz --out $R/$n.jsonl"
  if [ -n "$DRY" ]; then
    win $n; holder $n
    st "DRY $n arm=$arm: cd $T && PYTHONPATH=$T timeout -s TERM $LEGTMO $cmd"
    st "$n rc=0 img=$(basename $img) blocks=$b host=anx-glx chip=$CHIPS sha=$SHA aiclk=000..000 samples=0 ms_device= digest= (DRY)"
    return 0
  fi
  # D34.3: the slot is the order, the window check is the box's, the flock is the mutex. slotwait waits for a later slot
  # of this row on the pair by itself; rc 5 = no slot left (a note to the orchestrator, the chain ends)
  $HOME/flx/slotwait.sh kernel $PAIR $EST >> $L/chainK.slotwait 2>&1; rc=$?
  [ $rc != 0 ] && { st "$n not started: slotwait kernel $PAIR $EST rc=$rc (5 = no slot of flx-kernel left on $PAIR, 3 = HANDBACK, 4 = gave up)"; exit 6; }
  $HOME/flx/winwait.sh $EST ${CHIPS//,/ } >> $L/chainK.winwait 2>&1; rc=$?
  [ $rc != 0 ] && { st "$n not started: winwait rc=$rc"; exit 6; }
  while :; do
    [ -e $HOME/fleet/HANDBACK ] && { st "$n refused: HANDBACK"; exit 3; }
    [ $(now) -gt $GIVEUP ] && { st "$n: gave up"; exit 5; }
    e=$(bound)
    [ $END != 0 ] && [ $(( $(now) + EST )) -gt $END ] && { st "$n not started: $EST s would not end by $(date -u -d @$END +%T)Z"; return 9; }
    if [ $e = 0 ] || [ $(( $(now) + EST )) -le $e ]; then   # else a booked window is too near: wait for it to pass
      if winq && win $n && holder $n && flock -w 60 9; then
        if closed $n 60; then
          e=$(bound); { [ $e = 0 ] || [ $(( $(now) + EST )) -le $e ]; } && break
        fi
        flock -u 9
      fi
    fi
    sleep 20
  done
  tmo=$LEGTMO; [ $e != 0 ] && [ $(( e - $(now) )) -lt $tmo ] && tmo=$(( e - $(now) ))
  t0=$(now); sampler $L/$n.aiclk
  st "$n launch arm=$arm blocks=$b host=anx-glx chip=$CHIPS sha=$SHA n=$ITS img=$img $x (timeout $tmo s, a net only)"
  ( cd $T && PYTHONPATH=$T /usr/bin/time -f "process wall %e s maxrss %M KB" timeout -s TERM $tmo $cmd > $L/$n.log 2>&1 9>&- ); rc=$?
  kill $SAMP 2>/dev/null
  flock -u 9
  EST=$(( $(now) - t0 + 30 ))
  a=$(aiclk $L/$n.aiclk)
  ms=$(grep -a -o "\"ms_device\": \[[0-9.]*" $R/$n.jsonl 2>/dev/null | tail -n 1 | cut -d"[" -f2)
  S=$(ls $R/${n}_case_*state.npz 2>/dev/null | head -n 1)
  d=""; [ -n "$S" ] && d=$(dig $S 2>/dev/null); same=no; [ "$d" = "$CDIG" ] && same=yes
  grep -a "^phase " $L/$n.log > $L/$n.phase 2>/dev/null
  st "$n rc=$rc img=$(basename $img) blocks=$b host=anx-glx chip=$CHIPS sha=$SHA aiclk=$a ms_device=$ms digest=$d bitwise-vs-control=$same n=$ITS wall=$(( $(now) - t0 ))s $(grep -a "process wall" $L/$n.log 2>/dev/null | tail -n 1) | $(grep -a "Error\|error:\|FATAL\|Traceback\|abort" $L/$n.log | head -n 2 | cut -c1-200 | tr "\n" " ")"
  [ $same = yes ] && [ -n "$S" ] && rm -f $S          # a state equal to the control's is not kept (1 GB each); one that differs stays for the grade
  case $rc in 124|137|143) st "$n was ENDED BY THE TIMEOUT ($tmo s): chain stops, chips $CHIPS need an open check before their next user (D30)"; exit 4;; esac
  return $rc
}

st "chainK armed pid $$ chips $CHIPS lock $LOCK tree $T ($SHA) its $ITS start $(date -u -d @$START +%T)Z end $([ $END = 0 ] && echo none || date -u -d @$END +%T)Z | A $AIMG [$AX] | B $BIMG [$BX] | X $XIMG [$XX]"
[ -d $T/ttfluid ] || { st "no tree at $T: not started"; exit 2; }
[ -e $TT_MESH_GRAPH_DESC_PATH ] || { st "no 1x2 descriptor: not started"; exit 2; }
while [ $(now) -lt $START ]; do sleep 5; done
for l in $LEGS; do
  n=${l%%:*}; arm=${l##*:}
  leg sk_$n $arm; r=$?
  [ $r = 9 ] && break
  if [ $arm != X ] && [ $r != 0 ]; then st "sk_$n failed (rc $r): chain ends, logs/sk_$n.log"; break; fi
done
st "chainK done: $(python3 $R/absum.py $ST sk 2>&1 | tail -n 1 | cut -c1-300)"
