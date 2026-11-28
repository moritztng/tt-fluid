#!/bin/bash
# flx-kernel skarm (anx-glx, no device by itself): arm the Small k2 A/B (run/chainK.sh) once the body image skB is
# lowered. chainK asks ~/flx/slotwait.sh kernel c45, ~/flx/winwait.sh and the pair's flock in front of every leg, so
# arming early is safe; an image that comes too late for the slot ends in slotwait's rc 5. Status: logs/skarm.status
K=$HOME/flx/kernel; L=$K/logs; st() { echo "$(date -u +%FT%TZ) $*" >> $L/skarm.status; }
END=$(date -u -d ${SKARM_END:-2026-11-28T06:00:00Z} +%s)
st "skarm start pid $$: waits for run/img/skB.ready, then arms chainK on chips ${CHIPS:-4,5} (slot c45 05:52Z-06:04Z in ~/flx/orch/SLOTS)"
while [ ! -e $K/run/img/skB.ready ]; do
  [ -e $HOME/fleet/HANDBACK ] && { st "HANDBACK: ends, chainK not armed"; exit 3; }
  [ $(date +%s) -gt $END ] && { st "skB not ready by $(date -u -d @$END +%T)Z: ends, chainK not armed"; exit 5; }
  sleep 20
done
st "skB ready: arming chainK (LEGS ${LEGS:-B0:B A1:A B1:B})"
cd $K && TREE=$K/tree3 BIMG=$K/run/img/skB BX="--blktab 1" LEGS="${LEGS:-B0:B A1:A B1:B}" CHIPS=${CHIPS:-4,5} exec bash run/chainK.sh
