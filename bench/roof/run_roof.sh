#!/bin/bash
# Roof probe on one chip of a shared box, with the clock record taken outside the process.
# Usage: run_roof.sh <TT_VISIBLE_DEVICES chip> <sysfs node> <tag> [roof_dm.py args]. Env: PY, DESC (mesh descriptor), L (log dir), TMO.
CHIP=$1; NODE=$2; TAG=$3; shift 3
H=$(cd "$(dirname "$0")" && pwd); PY=${PY:-$HOME/fleet/env/bin/python}; L=${L:-$H/logs}; mkdir -p $L
[ -e $HOME/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
[ -n "$(cat /proc/driver/tenstorrent/$NODE/pids 2>/dev/null)" ] && { echo "node $NODE busy"; exit 4; }
: ${DESC:=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / 'tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto')" 2>/dev/null | tail -1)}
export TT_VISIBLE_DEVICES=$CHIP TT_MESH_GRAPH_DESC_PATH=$DESC
( while :; do echo "$(date +%s.%N) $(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk)"; sleep 0.05; done ) > $L/$TAG.aiclk 2>&1 & CLK=$!
trap "kill $CLK 2>/dev/null" EXIT
echo "$(date -u +%FT%TZ) start chip $CHIP node $NODE load $(cut -d' ' -f1-3 /proc/loadavg)" > $L/$TAG.log
timeout -s TERM ${TMO:-900} $PY -u $H/roof_dm.py --node $NODE --aiclk-log $L/$TAG.aiclk --out $L/$TAG.jsonl "$@" >> $L/$TAG.log 2>&1
RC=$?
echo "$(date -u +%FT%TZ) rc $RC load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L/$TAG.log
exit $RC
