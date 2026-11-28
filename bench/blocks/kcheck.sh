#!/bin/bash
shopt -s extglob
# flx-kernel kcheck (anx-glx, CPU only, no device, no ttnn import): syntax and semantics of vm_compute.cpp for the three
# TRISC builds with the installed SFPI compiler and tt-metal headers, before a device leg has to find a compile error.
#   kcheck.sh TREE DEFINES_FILE [EXTRA_DEFINE ...]     e.g. kcheck.sh ~/flx/kernel/tree4 defs.h CFD_PROFW=1
# The command is rebuilt from the include directories of a cached JIT build (trisck.o.dephash); it is validated by
# compiling a tree whose device build is known good (same script, no extra define) before it is believed.
T=${1:?tree}; DEFS=${2:?defines_generated.h}; shift 2
A=$HOME/fleet/env/lib/python3.10/site-packages/ttnn; M=$A/tt_metal; B=$A/runtime/sfpi/compiler/bin
W=$(mktemp -d /tmp/kcheck.XXXXXX); trap "rm -rf $W" EXIT
SRCD=$(ls -dt $HOME/.cache/tt-metal-cache/*/kernels/vm_compute/*/ | head -n 1)
cp $SRCD/chlkc_descriptors.h $W/ 2>/dev/null
cp $DEFS $W/defines_generated.h
for d in "$@"; do echo "#define ${d%%=*} ${d#*=}" >> $W/defines_generated.h; done
for n in unpack math pack isolate_sfpu; do
  N=$(echo $n | tr a-z A-Z)
  printf '#define TRISC_%s\n#include "defines_generated.h"\n#include "%s/ttfluid/device/kernels/vm_compute.cpp"\n' "$N" "$T" > $W/chlkc_$n.cpp
done
INC="-I$W -I$A -I$M -I$M/api -I$M/api/tt-metalium -I$M/hostdevcommon/api -I$M/hw/inc -I$M/hw/inc/api -I$M/hw/inc/internal -I$M/hw/inc/internal/tt-1xx -I$M/hw/inc/internal/tt-1xx/blackhole -I$M/hw/inc/internal/tt-1xx/blackhole/noc -I$M/hw/inc/hostdev -I$M/hw/ckernels/blackhole/metal/common -I$M/hw/ckernels/blackhole/metal/llk_io -I$M/hw/ckernels/blackhole/metal/llk_api -I$M/hw/ckernels/blackhole/metal/llk_api/llk_sfpu -I$M/hw/firmware/src/tt-1xx -I$M/third_party/tt_llk -I$M/third_party/tt_llk/common -I$M/third_party/tt_llk/tt_llk_blackhole/common/inc -I$M/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu -I$M/third_party/tt_llk/tt_llk_blackhole/llk_lib -I$A/runtime/sfpi/include -I$M/tools/profiler -I$T/ttfluid/device/kernels $KCHECK_INC"
FL="-mcpu=tt-bh-tensix -std=c++17 -ffast-math -fno-exceptions -O3 -fsyntax-only -fno-use-cxa-atexit -Wall -Werror -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function"
DEF="-DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DKERNEL_BUILD -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140 -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DKERNEL_COMPILE_TIME_ARGS=${KCHECK_CTA:-0,1,2,3,4,5,6,7,8,9,10,11,12,16,14,15,16,17,18,19,20,21,22,23} $KCHECK_DEF"
rc=0
i=0
for n in UNPACK MATH PACK; do
  l=$(echo $n | tr A-Z a-z)
  $B/riscv-tt-elf-g++ $FL $DEF -DCOMPILE_FOR_TRISC=$i -DUCK_CHLKC_$n -DNAMESPACE=chlkc_$l -DPROCESSOR_INDEX=$((i + 2)) $INC $M/hw/firmware/src/tt-1xx/trisck.cc > $W/$l.log 2>&1; r=$?
  echo "kcheck $n rc=$r errors=$(grep -c "error" $W/$l.log)"
  [ $r != 0 ] && { rc=1; grep -a "error\|Error" $W/$l.log | head -n ${KCHECK_N:-8} | cut -c1-330; }
  # KCHECK_SIZE=1: the object's text bytes at -O3 without LTO, a proxy for the kernel's share of the 69 KB kernel
  # config buffer (the device build links with LTO: compare two trees with it, do not read it as the linked size)
  [ -n "$KCHECK_SIZE" ] && [ $r = 0 ] && $B/riscv-tt-elf-g++ ${FL/-fsyntax-only/-c -o $W/$l.o} $DEF -DCOMPILE_FOR_TRISC=$i -DUCK_CHLKC_$n -DNAMESPACE=chlkc_$l -DPROCESSOR_INDEX=$((i + 2)) $INC $M/hw/firmware/src/tt-1xx/trisck.cc > /dev/null 2>&1 && echo "kcheck $n text+data bytes (no LTO): $($B/riscv-tt-elf-size $W/$l.o | tail -n 1 | awk '{print $1 + $2}')"
  i=$((i + 1))
done
# the reader (vm_ncrisc.cpp) on NCRISC: its defines arrive on the command line in a device build; here the compute
# kernel's file plus the reader's own (KCHECK_NDEF), as run_mesh adds them for --tsort 1 --bgts 5 --tgts 2
printf '#include "%s/ttfluid/device/kernels/vm_ncrisc.cpp"\n' "$T" > $W/kernel_includes.hpp
NA=$(printf '2,%.0s' $(seq 1 199))2   # every argument 2: a CB id, a power of two, and a DRAM interleaved TensorAccessorArgs word
$B/riscv-tt-elf-g++ ${FL/tt-bh-tensix/tt-bh} ${DEF/-DKERNEL_COMPILE_TIME_ARGS=*([^ ])/} -DKERNEL_COMPILE_TIME_ARGS=${KCHECK_NCTA:-$NA} -DCOMPILE_FOR_NCRISC -DPROCESSOR_INDEX=1 -DNOC_INDEX=0 -DPCIE_NOC_X=11 -DPCIE_NOC_Y=0 -DDISPATCH_MESSAGE_ADDR=0xFFB70438 -include $W/defines_generated.h ${KCHECK_NDEF:--DCFD_TSORT=1 -DCFD_BGTS=5} $INC $M/hw/firmware/src/tt-1xx/ncrisck.cc > $W/ncrisc.log 2>&1; r=$?
echo "kcheck NCRISC rc=$r errors=$(grep -c "error" $W/ncrisc.log)"
[ $r != 0 ] && { rc=1; grep -a "error\|Error" $W/ncrisc.log | head -n ${KCHECK_N:-8} | cut -c1-330; }
exit $rc
