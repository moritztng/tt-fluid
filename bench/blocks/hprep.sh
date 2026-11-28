#!/bin/bash
# flx-kernel hprep, CPU only: hood k1 images (hood65 on one chip, T 6) of the block A/B. Inline body programs make the
# level-fused tile loop span 64 code pages, so both arms run with a 128-page code ring (--page must be a power of 2); without --levels 32.
K=$HOME/flx/kernel; T=$K/tree; R=$K/run; L=$K/logs; PY=$HOME/fleet/env/bin/python
FL="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --tsort 1 --l1 1"
export PYTHONPATH=$T PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=26
st() { echo "$(date -u +%FT%TZ) $*" >> $L/prep.status; }
mkdir -p $R/img; cd $T
img() {
  [ -e $R/img/$1.ready ] && return 0
  rm -rf $R/img/$1
  nice -n 19 /usr/bin/time -f "TIME %e s maxrss %M KB" timeout 2400 $PY -u -m ttfluid.device.run_mesh $2 --blocks $3 --images $R/img/$1 > $L/img_$1.out 2>&1
  local rc=$?
  [ $rc = 0 ] && [ -e $R/img/$1/digests.json ] && touch $R/img/$1.ready
  st "img $1 rc=$rc $(grep -a "^blocks:" $L/img_$1.out | cut -c1-200) | $(grep -a "AssertionError\|^TIME" $L/img_$1.out | tr "\n" " " | cut -c1-220) | $(du -sh $R/img/$1 2>/dev/null | cut -f1)"
}
HD="case 20 $FL --levels 1 --page 128 --case $HOME/flx/asm/hood65 --min-free-gb 100 --mcache $R/hk1_lv1_p128.pkl.zst"
HD0="case 20 $FL --page 32 --case $HOME/flx/asm/hood65 --min-free-gb 100 --mcache $R/hk1_p32.pkl.zst"
st "hprep start pid $$ tree $(cat $K/tree.sha)"
img hB "$HD" 1; img hA "$HD" 0; img hX "$HD" 2
img h0B "$HD0" 1; img h0A "$HD0" 0; img h0X "$HD0" 2
st "hprep done"
