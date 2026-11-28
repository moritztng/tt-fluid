# flx-asm: shared checks of the post-window device chains (exp7: Small k2 on a pair, exp8: hood k1 on one chip).
# The caller sets ST (status file), CHIPS (comma list; chip id = sysfs node for chips 0-15) and LOCK (the chips' flock file).
PY=$HOME/fleet/env/bin/python
st() { echo "$(date -u +%FT%TZ) $*" >> $ST; }
now() { date -u +%s; }
win() { # NAME: 0 = no booked window of another row now or within 9 min, and the mirror of pc's booking file is fresh
  local t=$(now) s slug a b rest ta tb
  s=$(date -u -d "$(cat $HOME/fleet/HOLDER.windows.stamp 2>/dev/null)" +%s 2>/dev/null)
  if [ -z "$s" ] || [ $((t - s)) -gt 1800 ]; then st "$1 refused: HOLDER.windows stamp missing or older than 30 min"; return 1; fi
  while read -r slug a b rest; do
    [ -z "$slug" ] && continue
    case "$slug" in \#*|flx-kernel) continue;; esac
    ta=$(date -u -d "$a" +%s 2>/dev/null); tb=$(date -u -d "$b" +%s 2>/dev/null)
    if [ -z "$ta" ] || [ -z "$tb" ]; then st "$1 refused: unreadable window line: $slug $a $b"; return 1; fi
    if [ $((t + 540)) -ge "$ta" ] && [ "$t" -le "$tb" ]; then st "$1 refused: window $slug $a $b"; return 1; fi
  done < $HOME/fleet/HOLDER.windows
  return 0
}
holder() { # NAME: 0 = every chip of CHIPS sits on a line of the HOLDER mirror whose holder starts with flx
  local c h
  for c in ${CHIPS//,/ }; do
    h=$(awk -v c=$c '$1 ~ /^[0-9]+(-[0-9]+)?$/ {n = split($1, r, "-"); lo = r[1] + 0; hi = (n > 1) ? r[2] + 0 : lo; if (c + 0 >= lo && c + 0 <= hi) {print $2; exit}}' $HOME/fleet/HOLDER 2>/dev/null)
    case "$h" in flx*) ;; *) st "$1 refused: HOLDER gives chip $c to [$h], not to an flx row"; return 1;; esac
  done
  return 0
}
held() { local n; for n in ${CHIPS//,/ }; do cat /proc/driver/tenstorrent/$n/pids 2>/dev/null; done | sort -u | tr "\n" " "; }
closed() { # NAME WAIT_S: 0 = no process holds a node of CHIPS (waits up to WAIT_S for that) and none takes one in the next 4 s
  local t=$(( $(now) + $2 ))
  while [ -n "$(held)" ] && [ $(now) -lt $t ]; do sleep 2; done
  [ -n "$(held)" ] && { st "$1 refused: node(s) of $CHIPS still held by pid(s) $(held)"; return 1; }
  sleep 4
  [ -n "$(held)" ] && { st "$1 refused: a node of $CHIPS was taken during the settle wait by pid(s) $(held)"; return 1; }
  return 0
}
sampler() { # FILE: sysfs AICLK of every node of CHIPS every 2 s, until killed (pid in SAMP). It does not keep the flock fd.
  ( exec 9>&-; while :; do echo "$(date -u +%T) load:$(cut -d" " -f1 /proc/loadavg) $(for n in ${CHIPS//,/ }; do echo -n "n$n:$(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk) "; done)"; sleep 2; done ) > $1 2>&1 &
  SAMP=$!
}
aiclk() { # FILE -> "<min>..<max> samples=<N>" over the samples after every node reached 1300 MHz, without the last (device closing)
  awk '{m = 1e9; M = 0; for (i = 3; i <= NF; i++) {split($i, x, ":"); v = x[2] + 0; if (v < m) m = v; if (v > M) M = v}
        if (f) {a[++k] = m; b[k] = M}
        if (m >= 1300) f = 1}
       END {mn = 1e9; mx = 0; for (i = 1; i < k; i++) {if (a[i] < mn) mn = a[i]; if (b[i] > mx) mx = b[i]}
            if (k < 2) printf "000..000 samples=0"; else printf "%d..%d samples=%d", mn, mx, k - 1}' $1
}
dig() { # STATE.npz -> its sha256 state digest (flx-sol's tool), empty when it cannot be read
  [ -e $1 ] && $PY $HOME/flx/sol/tools/statedig.py $1 2>&1 | grep -o "[0-9a-f]\{64\}" | tail -n 1
}
