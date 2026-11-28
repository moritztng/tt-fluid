#!/bin/bash
# flx-kernel t3test (anx-glx, CPU only, under 2 GB): tree3 (the L1 budget counts the body table) builds the same pitzDaily
# table image as tree2 did (img/pzT): equal digests.json and blk_tab.json. Status: logs/t3test.status
K=$HOME/flx/kernel; T=$K/tree3; R=$K/run; L=$K/logs; ST=$L/t3test.status; PY=$HOME/fleet/env/bin/python
FLP="--arch bh --mesh 1,1 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --l1 1"
export PYTHONPATH=$T PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=26
I=$R/img/pzT3.$$
( cd $T && nice -n 19 /usr/bin/time -f "TIME %e s maxrss %M KB" timeout 900 $PY -u -m ttfluid.device.run_mesh case 50 $FLP --levels 1 --case $K/tree/cases/pitzDaily \
    --min-free-gb 100 --mcache $R/pz50.pkl --blocks 1 --blktab 1 --images $I > $L/img_pzT3.out 2>&1 ); rc=$?
d=no; cmp -s $I/digests.json $R/img/pzT/digests.json && d=yes
t=no; cmp -s $I/blk_tab.json $R/img/pzT/blk_tab.json && t=yes
echo "$(date -u +%FT%TZ) t3test rc=$rc tree3 $(cat $K/tree3.sha) image-digests-equal-tree2=$d table-equal=$t | $(grep -a "Error\|^TIME" $L/img_pzT3.out | tr "\n" " " | cut -c1-200)" >> $ST
