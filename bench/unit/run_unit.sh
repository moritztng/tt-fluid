#!/bin/bash
# Unit-cost bench on one chip of a shared box, with the clock record taken outside the process (the Python sampler
# cannot run while ttnn blocks). Usage: run_unit.sh <chip> <tag> [unit_costs.py args]
# <chip> is the TT_VISIBLE_DEVICES number (the one booking files use). The /dev/tenstorrent node differs on a Galaxy:
# NODE=<n> overrides the anx-glx rule below, and unit_costs.py refuses to run if the node it opened is not NODE.
# Writes logs/<tag>.json and logs/<tag>.aiclk next to this script's parent. PY defaults to the fleet venv.
# The box needs blk_exec.h (ttfluid/device/kernels) copied next to ub_compute.cpp.
CHIP=$1; TAG=$2; shift 2
B=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/fleet/env/bin/python}; L=$B/logs; mkdir -p $L
[ -e $HOME/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
NODE=${NODE:-$(( CHIP < 16 ? CHIP : (CHIP < 24 ? CHIP + 8 : CHIP - 8) ))}
# NOTAFTER=<epoch>: a chip shared with another row's chain. Take the chip's two lock files (~/spd/locks/chip<id> and
# chip<id>.lock, the convention of spd, bc4, flx-pfuse and flx-gather's chains: a leg holds both while it runs), wait
# for an empty device node, and refuse to start at or after NOTAFTER. The locks are held until this script exits.
if [ -n "$NOTAFTER" ]; then
  LK=$HOME/spd/locks; mkdir -p $LK; exec 9>>$LK/chip$CHIP 8>>$LK/chip$CHIP.lock
  left() { echo $(( NOTAFTER - $(date +%s) )); }
  { [ $(left) -gt 0 ] && flock -w $(left) 9 && [ $(left) -gt 0 ] && flock -w $(left) 8; } || { echo "$(date -u +%T) no chip lock before NOTAFTER"; exit 5; }
  while [ -n "$(cat /proc/driver/tenstorrent/$NODE/pids 2>/dev/null)" ] && [ $(left) -gt 0 ]; do sleep 1; done
  [ $(left) -gt 0 ] || { echo "$(date -u +%T) past NOTAFTER"; exit 5; }
  echo "$(date -u +%T) chip locks held, node $NODE empty, $(left) s to NOTAFTER"
fi
[ -n "$(fuser /dev/tenstorrent/$NODE 2>/dev/null)" ] && { echo "chip $CHIP (node $NODE) busy"; exit 4; }
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP TT_BIO_LEASE_HOLDER=worker:flx-kernel
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / 'tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto')")
echo "$(date -u +%FT%TZ) start chip $CHIP node $NODE load $(cut -d' ' -f1-3 /proc/loadavg)"
( while :; do echo "$(date +%s.%N) $(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk)"; sleep 0.02; done ) > $L/$TAG.aiclk 2>&1 & CLK=$!
timeout 420 $PY -u $B/unit/unit_costs.py --sysfs-node $NODE --json $L/$TAG.json "$@" 2>&1 | grep -v "| info \|| warning \|^Total nodes\|^Degree hist\|topology_solver\|^=== "
RC=${PIPESTATUS[0]}; kill $CLK 2>/dev/null
$PY - $L/$TAG.json $L/$TAG.aiclk <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
t0, t1 = j["runs"][1]["t0"], j["runs"][-1]["t1"]
v = [int(l.split()[1]) for l in open(sys.argv[2]) if len(l.split()) == 2 and t0 <= float(l.split()[0]) <= t1]
print("AICLK-SYSFS min %s max %s MHz, %d samples inside the kept runs (%.2f s)" % (min(v, default=None), max(v, default=None), len(v), t1 - t0))
j["aiclk_sysfs"] = dict(min=min(v, default=None), max=max(v, default=None), n=len(v))
json.dump(j, open(sys.argv[1], "w"), indent=1)
PYEOF
echo "$(date -u +%FT%TZ) rc $RC load $(cut -d' ' -f1-3 /proc/loadavg)"
