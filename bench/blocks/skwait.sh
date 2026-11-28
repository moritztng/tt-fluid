#!/bin/bash
# flx-kernel skwait (anx-glx): start the Small k2 body image (table form, tree3) the minute the whole-box window ends.
# The orchestrator's RAM guard stops FLX lowerings from 03:40Z until "win2 glx2 exit" in ~/flx/galaxy/logs/run.status
# (note hang0310), so nothing here asks for a lowering slot before that line exists, or before 04:31Z without it.
# Then: SKTREE=tree3 skprep.sh skB 1 "--blktab 1 --l1-bytes 1461760" (one lowering under ~/flx/lowerlock.sh).
# --l1-bytes 1461760 is the real CB region of a Galaxy Blackhole core (all of L1 but the 69 KB kernel config buffer);
# the fit check counts the table since a4bc687, and the 1,370,000 default could refuse the image after the lowering.
K=$HOME/flx/kernel; L=$K/logs; ST=$L/skprep.status; RS=$HOME/flx/galaxy/logs/run.status
LATEST=$(date -u -d 2026-11-28T04:31:00Z +%s); GIVEUP=$(date -u -d 2026-11-28T07:00:00Z +%s)
echo "$(date -u +%FT%TZ) skwait armed pid $$: skB (table form, tree3 $(cat $K/tree3.sha)) lowers after the window" >> $ST
while :; do
  [ -e $HOME/fleet/HANDBACK ] && { echo "$(date -u +%FT%TZ) skwait ends: HANDBACK" >> $ST; exit 3; }
  now=$(date +%s); [ $now -ge $GIVEUP ] && { echo "$(date -u +%FT%TZ) skwait gave up" >> $ST; exit 4; }
  grep -q "win2 glx2 exit" $RS 2>/dev/null && { why="win2 glx2 exit in run.status"; break; }
  [ $now -ge $LATEST ] && { why="04:31Z passed without an exit line"; break; }
  sleep 20
done
echo "$(date -u +%FT%TZ) skwait: go ($why)" >> $ST
cd $K && SKTREE=$K/tree3 exec ./run/skprep.sh skB 1 "--blktab 1 --l1-bytes 1461760"
