#!/bin/bash
# flx-kernel skarm (anx-glx, no device by itself): arm the Small k2 A/B (run/chainK.sh) once the body image skB is
# lowered AND the slot file (~/flx/orch/SLOTS, the copy of pc state/flx/SLOTS) holds a slot of this row that has not
# ended. The pair is read from that line, so a re-planned slot needs no re-arm. chainK asks ~/flx/slotwait.sh,
# ~/flx/winwait.sh and the pair's flock in front of every leg. When all LEGS ran rc 0 the image is removed if it lives
# under /dev/shm (RAM). Status: logs/skarm.status
#   env SKARM_END (UTC stamp, required): gives up then.  LEGS (default "B0:B A1:A B1:B"), TREE, BX as chainK's.
K=$HOME/flx/kernel; L=$K/logs; S=$HOME/flx/orch/SLOTS; I=$K/run/img/skB
st() { echo "$(date -u +%FT%TZ) $*" >> $L/skarm.status; }
END=$(date -u -d "${SKARM_END:?a UTC stamp}" +%s) || exit 2
LEGS=${LEGS:-B0:B A1:A B1:B}
slot() { # -> the pair (c01, c45, c67) of this row's first slot that ends 240 s or more from now; empty if none
  local pair a b row rest tb
  while read -r pair a b row rest; do
    case "$pair" in c[0-9][0-9]) ;; *) continue;; esac
    [ "$row" = kernel ] || continue
    tb=$(date -u -d "$b" +%s 2>/dev/null) || continue
    [ $(( tb - $(date +%s) )) -ge 240 ] && { echo $pair; return; }
  done < $S
}
st "skarm start pid $$: waits for $I.ready and a kernel line in $S, gives up $(date -u -d @$END +%FT%TZ)"
said=""
while :; do
  [ -e $HOME/fleet/HANDBACK ] && { st "HANDBACK: ends, chainK not armed"; exit 3; }
  if [ $(date +%s) -gt $END ]; then
    T=$(readlink -f $I); r=$([ -e $I.ready ] && echo yes || echo no)
    case "$T" in /dev/shm/*) [ $r = yes ] && rm -rf "$T" $I $I.ready && r="yes, removed from RAM";; esac
    st "gave up: image ready=$r, slot=$(slot)"; exit 5
  fi
  if [ -e $I.ready ]; then
    p=$(slot)
    [ -n "$p" ] && break
    [ -z "$said" ] && { st "skB ready, no kernel slot in $S yet: waiting"; said=1; }
  fi
  sleep 20
done
p=${p#c}; CH=${p:0:1},${p:1:1}
st "skB ready and slot c$p open or ahead: arming chainK on chips $CH (LEGS $LEGS)"
cd $K && TREE=${TREE:-$K/tree3} BIMG=$I BX="${BX:---blktab 1}" LEGS="$LEGS" CHIPS=$CH GIVEUP=$(date -u -d @$END +%FT%TZ) bash run/chainK.sh
rc=$?
ok=1; for l in $LEGS; do grep -q " sk_${l%%:*} rc=0 " $L/chainK.status || ok=0; done
T=$(readlink -f $I)
if [ $ok = 1 ]; then
  case "$T" in /dev/shm/*) rm -rf "$T" $I $I.ready; st "chainK rc=$rc, every leg rc 0: image $T removed (RAM given back)";; *) st "chainK rc=$rc, every leg rc 0: image $T kept (on disk)";; esac
else
  st "chainK rc=$rc, legs missing: image $T KEPT for a re-arm ($(du -sh $T 2>/dev/null | cut -f1)); logs/chainK.status"
fi
