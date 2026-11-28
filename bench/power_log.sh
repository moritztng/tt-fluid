#!/bin/bash
# Passive power log for a Tenstorrent host during a run: per-chip board telemetry (hwmon power1_input), CPU package
# power (RAPL energy counter, needs sudo -n), AICLK and load, one line per interval. It opens no device and reads
# only sysfs, so it can run beside any workload. This is not a wall reading: RAM, drives, fans and PSU loss are not
# in it.
# Usage: power_log.sh <out.tsv> [interval-s (10)] [max-s (86400)]
set -u
OUT=$1; DT=${2:-10}; MAX=${3:-86400}; END=$(( $(date +%s) + MAX ))
R=/sys/class/powercap/intel-rapl:0/energy_uj
chips() {   # "<pci> <hwmon-dir> <class-dir>" per chip, in PCI order
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = blackhole ] || continue
    p=$(basename $(readlink -f $h/device))
    for c in /sys/class/tenstorrent/*; do [ "$(basename $(readlink -f $c/device))" = $p ] && echo "$p $h $c"; done
  done | sort
}
C=$(chips); N=$(echo "$C" | wc -l)
{ printf '# %s chips:' "$(hostname)"; echo "$C" | awk '{printf " %s=%s", $1, $3}'; echo
  echo "# epoch cpu_W load1 then per chip (PCI order): W aiclk"; } >> $OUT
e0=$(sudo -n cat $R 2>/dev/null); t0=$(date +%s.%N)
while [ $(date +%s) -lt $END ]; do
  sleep $DT
  e1=$(sudo -n cat $R 2>/dev/null); t1=$(date +%s.%N)
  line=$(awk -v a=${e0:-0} -v b=${e1:-0} -v ta=$t0 -v tb=$t1 'BEGIN{printf "%d %.1f", tb, (b>=a && a>0) ? (b-a)/1e6/(tb-ta) : -1}')
  line="$line $(cut -d' ' -f1 /proc/loadavg)"
  while read -r p h c; do
    line="$line $(awk '{printf "%.1f", $1/1e6}' $h/power1_input 2>/dev/null) $(cat $c/tt_aiclk 2>/dev/null)"
  done <<< "$C"
  echo "$line" >> $OUT; e0=$e1; t0=$t1
done
