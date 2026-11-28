#!/bin/bash
# flx-kernel pzprep, CPU only: pitzDaily images of the block A/B on one full chip (T 1, every tile in L1, so without the
# page-streaming flags of the hood runs). A = --blocks 0, B = 1, X = 2 (broken); pz* with --levels 1, pz0* without.
K=$HOME/flx/kernel; T=$K/tree; R=$K/run; L=$K/logs; PY=$HOME/fleet/env/bin/python
FLP="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --l1 1"
export PYTHONPATH=$T PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=26
st() { echo "$(date -u +%FT%TZ) $*" >> $L/prep.status; }
mkdir -p $R/img; cd $T
img() {
  [ -e $R/img/$1.ready ] && return 0
  rm -rf $R/img/$1
  nice -n 19 /usr/bin/time -f "TIME %e s maxrss %M KB" timeout 600 $PY -u -m ttfluid.device.run_mesh $2 --blocks $3 --images $R/img/$1 > $L/img_$1.out 2>&1
  local rc=$?
  [ $rc = 0 ] && [ -e $R/img/$1/digests.json ] && touch $R/img/$1.ready
  st "img $1 rc=$rc $(grep -a "^blocks:" $L/img_$1.out | cut -c1-200) | $(grep -a "^TIME" $L/img_$1.out | cut -c1-60) | $(du -sh $R/img/$1 2>/dev/null | cut -f1)"
}
PZ="case 50 $FLP --levels 1 --case $T/cases/pitzDaily --min-free-gb 100 --mcache $R/pz50.pkl"
PZ0="case 50 $FLP --case $T/cases/pitzDaily --min-free-gb 100 --mcache $R/pz50_l0.pkl"
img pzA "$PZ" 0; img pzB "$PZ" 1; img pzX "$PZ" 2
img pz0A "$PZ0" 0; img pz0B "$PZ0" 1; img pz0X "$PZ0" 2
st "pzprep done"
