#!/bin/bash
# Pack a built SPUMA tree into the runtime bundle layout the boxes extract at / (opt/spumaM/{etc,bin,platforms} +
# opt/nvrt, the NVHPC runtime libs and CUDA compat libcuda taken from an existing bundle), after checking that every
# CUDA-carrying library has a cubin for each requested arch.
# Usage: spuma_pack.sh <tree> <old-bundle.tar.zst> <out.tar.zst> "<sm list, e.g. 80 90 100>" <cuobjdump>
# env PREC (DP or SPDP, default DP) and NAME (the /opt dir, default spumaM; an SPDP bundle goes to spumaS beside it).
set -eu
T=$1; OLD=$2; OUT=$3; SMS=$4; CO=$5
PREC=${PREC:-DP}; NAME=${NAME:-spumaM}
L=$T/platforms/linux64Nvidia${PREC}Int32Opt
[ -x $L/bin/simpleFoam ] || { echo "no simpleFoam in $L/bin"; exit 1; }
bad=0
for f in $L/lib/*.so $L/bin/simpleFoam; do
  have=$($CO --list-elf $f 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
  [ -z "$have" ] && continue
  for s in $SMS; do case " $have" in *" sm_$s "*) ;; *) echo "MISSING sm_$s in $f (has $have)"; bad=1;; esac; done
done
[ $bad = 0 ] || exit 1
W=$(mktemp -d); trap 'rm -rf $W' EXIT
zstd -dc $OLD | tar xf - -C $W opt/nvrt
mkdir -p $W/opt/$NAME && cp -a $T/etc $T/bin $W/opt/$NAME/ && mkdir -p $W/opt/$NAME/platforms && cp -a $L $W/opt/$NAME/platforms/
rm -rf $W/opt/$NAME/platforms/*/src $W/opt/$NAME/platforms/*/applications
tar cf - -C $W opt | zstd -T16 -10 -q > $OUT
ls -la $OUT
