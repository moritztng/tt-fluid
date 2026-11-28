#!/bin/bash
# flx-kernel pzt: the table form of the bodies (--blktab 1) against stock words and against the inline form, pitzDaily
# on one chip (T 1, 50 its). Chip 26 (node 18) by turns: both of flx-gather's chip locks per leg (as its chainG2.sh),
# no HANDBACK, no window within 9 min, node closed. Series tt: A = stock words, B = bodies from the kernel's table,
# X = broken arm (table form); ti_B*: bodies inline (the form measured on chip 2). Status: logs/pzt.status
K=$HOME/flx/kernel; T=$K/tree2; R=$K/run; L=$K/logs; ST=$L/pzt.status
. $R/ablib.sh
SHA=$(cat $K/tree2.sha); CHIP=26; NODE=18; CHIPS=$NODE; LK=$HOME/spd/locks
END=$(date -u -d ${END:-2026-11-28T03:40:00Z} +%s)
FLP="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --l1 1"
PZ="case 50 $FLP --levels 1 --case $K/tree/cases/pitzDaily"
export PYTHONPATH=$T PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CHIP
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto\")" 2>/dev/null | tail -n 1)
st "pzt start pid $$ tree2 $SHA chip $CHIP node $NODE"
img() { # NAME BLOCKS BLKTAB
  local rc
  [ -e $R/img/$1.ready ] && return 0
  [ -d $R/img/$1 ] && mv $R/img/$1 $R/img/$1.old.$$
  ( cd $T && nice -n 19 /usr/bin/time -f "TIME %e s maxrss %M KB" timeout 600 $PY -u -m ttfluid.device.run_mesh $PZ --min-free-gb 100 --mcache $R/pz50.pkl --blocks $2 --blktab $3 --images $R/img/$1 > $L/img_$1.out 2>&1 ); rc=$?
  [ $rc = 0 ] && [ -e $R/img/$1/digests.json ] && touch $R/img/$1.ready
  st "img $1 rc=$rc $(grep -a "^blocks:" $L/img_$1.out | cut -c1-260) | $(grep -a "AssertionError\|Error\|^TIME" $L/img_$1.out | tr "\n" " " | cut -c1-220) | $(du -sh $R/img/$1 2>/dev/null | cut -f1) table $(wc -c < $R/img/$1/blk_tab.json 2>/dev/null) B json"
  return $rc
}
leg() { # NAME IMG BLOCKS BLKTAB
  local n=$1
  (
    flock -w 300 9 || { st "$n: no lock chip$CHIP"; exit 98; }
    flock -w 300 8 || { st "$n: no lock chip$CHIP.lock"; exit 98; }
    [ -e $HOME/fleet/HANDBACK ] && { st "$n refused: HANDBACK"; exit 3; }
    [ $(( $(now) + 60 )) -gt $END ] && { st "$n not started: it would not end by $(date -u -d @$END +%T)Z"; exit 9; }
    win $n || exit 4
    ( CHIPS=$CHIP; holder $n ) || exit 4
    closed $n 60 || exit 4
    ( exec 9>&- 8>&-; while :; do echo "$(date -u +%T.%N | cut -c1-12) load:$(cut -d" " -f1 /proc/loadavg) n$NODE:$(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk)"; sleep 0.25; done ) > $L/$n.aiclk 2>&1 &
    SAMP=$!
    ( cd $T && timeout 150 $PY -u -m ttfluid.device.run_mesh $PZ --blocks $3 --blktab $4 --from-images $R/img/$2 --noemu --runs 1 --os 1 --sysfs-nodes $NODE --out $R/$n.jsonl > $L/$n.log 2>&1 9>&- 8>&- ); rc=$?
    kill $SAMP 2>/dev/null
    ms=$(grep -a -o "\"ms_device\": \[[0-9.]*" $R/$n.jsonl 2>/dev/null | tail -n 1 | cut -d"[" -f2)
    d=$(dig $(ls $R/${n}_case_*state.npz 2>/dev/null | head -n 1) 2>/dev/null)
    grep -a "^phase " $L/$n.log > $L/$n.phase 2>/dev/null
    st "$n rc=$rc img=$2 blocks=$3 host=anx-glx chip=$CHIP sha=$SHA aiclk=$(aiclk $L/$n.aiclk) ms_device=$ms digest=$d blktab=$4 | $(grep -a "blktab:\|Error\|error:\|FATAL\|Traceback\|abort" $L/$n.log | head -n 3 | cut -c1-220 | tr "\n" " ")"
    exit $rc
  ) 9>>$LK/chip$CHIP 8>>$LK/chip$CHIP.lock
}
img pzT 1 1 || { st "pzt: the table image failed, logs/img_pzT.out"; exit 1; }
img pzTX 2 1
leg tt_A0 pzA 0 0 || { st "pzt: arm A failed on tree2: chain ends"; exit 6; }
leg tt_B0 pzT 1 1 || { st "pzt: the table arm failed: chain ends, logs/tt_B0.log"; exit 7; }
leg tt_A1 pzA 0 0; leg tt_B1 pzT 1 1; leg ti_B1 pzB 1 0
leg tt_A2 pzA 0 0; leg tt_B2 pzT 1 1; leg ti_B2 pzB 1 0
leg tt_X pzTX 2 1
st "pzt done"
