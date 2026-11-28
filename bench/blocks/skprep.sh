#!/bin/bash
# flx-kernel skprep (anx-glx, CPU only): Small k2 images of the block A/B. The kept control's lowering flags (flx-sol
# tools/lower2.sh: main c22145e, `case 1000`, no machine cache) from a SNAPSHOT of the kernel tree, each lowering under
# ~/flx/lowerlock.sh (four slots box-wide, starts only with 110 GB of RAM available). One lowering is 84 GB and about
# 18 minutes; a machine cache is not written (flx-asm's cost 129 GB maxrss, 40 minutes and 56 GB of disk).
# Usage: skprep.sh NAME BLOCKS EXTRA [NAME BLOCKS EXTRA ...]   EXTRA = extra lowering flags in one word list, `none` for none
#   e.g. skprep.sh skB_inline 1 "--page 32" skA 0 none          (triples run one after the other: one lock slot at a time)
#   -> run/img/NAME (+ NAME.ready), log logs/sk_low_NAME.out, status lines in logs/skprep.status
# env SKTREE: source tree (default run/sktree, its sha in run/sktree.sha). The tree must not change under a running
#   lowering: encode.py imports blockgen and blklower only when the first body is encoded.
# env SKFLOOR: --min-free-gb of the image write (default 70, DECISIONS D37.8; the write needs the floor + 29 GB free).
# env SKROOT: directory the image is written under, with run/img/NAME as a link to it. /dev/shm/flx-kernel keeps a
#   Small k2 image (13 GB) in RAM when / is under the floor: delete it after its legs.
# BLOCKS: 0 stock op words (arm A; its digests.json must equal the control's), 1 generated bodies (arm B), 2 broken arm X.
K=$HOME/flx/kernel; R=$K/run; L=$K/logs; SB=$HOME/flx/sol; PY=$HOME/fleet/env/bin/python; T=${SKTREE:-$R/sktree}
CTL=$SB/img/small_k2_jac
FL="--arch bh --mesh 1,2 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --forces $SB/case/small_fc_controlDict:forceCoeffsAll --case $SB/case/small_tt"
export PYTHONPATH=$T PYTHONHASHSEED=0 OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=26
st() { echo "$(date -u +%FT%TZ) $*" >> $L/skprep.status; }
mkdir -p $R/img $L
img() { # NAME BLOCKS EXTRA
  local n=$1 b=$2 x=$3 I=$R/img/$1 rc t0 same lp
  [ "$x" = none ] && x=""
  case "$n" in sk*) ;; *) st "img $n refused: the name must start with sk"; return 2;; esac
  [ -e $I.ready ] && { st "img $n: ready already"; return 0; }
  [ -e $HOME/fleet/HANDBACK ] && { st "img $n refused: HANDBACK"; return 3; }
  [ -d $T/ttfluid ] || { st "img $n refused: no tree at $T"; return 2; }
  if [ -n "$SKROOT" ]; then   # the image lives under SKROOT, run/img/NAME is a link to it
    if [ -e $I ] && [ ! -L $I ]; then st "img $n refused: $I exists and is not a link"; return 2; fi
    mkdir -p $SKROOT/$n && ln -sfn $SKROOT/$n $I || { st "img $n refused: no directory under $SKROOT"; return 2; }
  fi
  st "img $n: lowering, pid $$ tree $T ($(cat $T.sha 2>/dev/null)) blocks $b flags [$x], waits for a lowerlock slot and 110 GB"
  t0=$(date +%s)
  ( cd $T && nice -n 10 /usr/bin/time -f "TIME %e s maxrss %M KB" $HOME/flx/lowerlock.sh timeout -s TERM 14400 \
      $PY -u -m ttfluid.device.run_mesh case 1000 $FL --tsort 1 --l1 1 $x --blocks $b --min-free-gb ${SKFLOOR:-70} --images $I > $L/sk_low_$n.out 2>&1 ); rc=$?
  find $I/ -name "*.npy" -exec fallocate -d {} \; 2>/dev/null
  [ $rc = 0 ] && [ -s $I/digests.json ] && touch $I.ready
  same="-"; if [ -s $I/digests.json ]; then cmp -s $I/digests.json $CTL/digests.json && same=yes || same=no; fi
  lp=$(grep -a -o "['\"]loop_pages['\"]: [0-9]*" $L/sk_low_$n.out | tail -n 1 | grep -o "[0-9]*$")
  st "img $n rc=$rc in $(( $(date +%s) - t0 )) s (lock wait included) $(du -sh $I/ 2>/dev/null | cut -f1) loop_pages=$lp image-digests-equal-control=$same | $(grep -a "^blocks:" $L/sk_low_$n.out | tail -n 1 | cut -c1-220) | $(grep -a "AssertionError\|tile loop spans\|CB bytes\|^TIME\|Killed\|MemoryError" $L/sk_low_$n.out | tr "\n" " " | cut -c1-300)"
  return $rc
}
st "skprep start pid $$ tree $T ($(cat $T.sha 2>/dev/null)): $*"
while [ $# -ge 3 ]; do img "$1" "$2" "$3"; shift 3; done
st "skprep done pid $$"
