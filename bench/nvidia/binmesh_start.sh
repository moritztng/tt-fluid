#!/bin/bash
# What does an ASCII mesh cost the GPU case at start? Usage: binmesh_start.sh <case-dir> <work-dir> Stock OpenFOAM v2412 simpleFoam (the code SPUMA
# forks; mesh and field read run on the host on both), one core, endTime 0: start, read mesh and fields, build the
# turbulence model and wall distance, exit before SIMPLE it 1. Arm A = the staged case as shipped (ASCII),
# arm B = the same case after `foamFormatConvert -constant` with writeFormat binary. Each arm twice (second = warm
# page cache, which is the state on a rented box right after unpacking). Real copies: the source files are hardlinked.
set +u
SRC=${1:?case dir} W=${2:?work dir}; mkdir -p $W; cd $W
source ${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
st() { echo "$(date -u +%FT%TZ) $*" >> $W/status; }
prep() { rm -rf $1; mkdir $1; cp -r --no-preserve=links $SRC/system $SRC/constant $SRC/0 $1/; foamDictionary -entry endTime -set 0 $1/system/controlDict > /dev/null; }
run() {   # run ARM REP: stamped log, wall from /usr/bin/time
  ( cd $1 && /usr/bin/time -f "%e s wall %U user %S sys %M KB" -o ../time_$1_$2.txt nice -n 5 simpleFoam 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > ../log_$1_$2.txt )
  st "$1 rep $2: $(cat time_$1_$2.txt | tr '\n' ' ')"; }
st "start pid $$ host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
prep A; prep B
foamDictionary -entry writeFormat -set binary B/system/controlDict > /dev/null
( cd B && /usr/bin/time -f "%e s" -o ../time_convert.txt foamFormatConvert -constant > ../log_convert.txt 2>&1 ); st "convert rc $? $(cat time_convert.txt) | $(head -c 400 B/constant/polyMesh/points | grep -a format | tr -s ' ') | mesh bytes A $(du -sb A/constant/polyMesh | cut -f1) B $(du -sb B/constant/polyMesh | cut -f1)"
for r in 1 2; do run A $r; run B $r; done
st "done"
