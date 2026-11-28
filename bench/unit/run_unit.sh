#!/bin/bash
# Unit-cost bench on one chip of a shared box, with the clock record taken outside the process (the Python sampler
# cannot run while ttnn blocks). Usage: run_unit.sh <chip = sysfs node> <tag> [unit_costs.py args]
# Writes logs/<tag>.json and logs/<tag>.aiclk next to this script's parent. PY defaults to the fleet venv.
CHIP=$1; TAG=$2; shift 2
B=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/fleet/env/bin/python}; L=$B/logs; mkdir -p $L
[ -e $HOME/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
[ -n "$(fuser /dev/tenstorrent/$CHIP 2>/dev/null)" ] && { echo "chip $CHIP busy"; exit 4; }
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP TT_BIO_LEASE_HOLDER=worker:flx-kernel
export TT_MESH_GRAPH_DESC_PATH=$($PY -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / 'tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto')")
echo "$(date -u +%FT%TZ) start chip $CHIP load $(cut -d' ' -f1-3 /proc/loadavg)"
( while :; do echo "$(date +%s.%N) $(cat /sys/class/tenstorrent/tenstorrent\!$CHIP/tt_aiclk)"; sleep 0.02; done ) > $L/$TAG.aiclk 2>&1 & CLK=$!
timeout 420 $PY -u $B/unit/unit_costs.py --sysfs-node $CHIP --json $L/$TAG.json "$@" 2>&1 | grep -v "| info \|| warning \|^Total nodes\|^Degree hist\|topology_solver\|^=== "
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
