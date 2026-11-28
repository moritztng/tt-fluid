#!/usr/bin/env python3
"""Pick <cores-per-GPU> idle physical cores for each of <n-GPUs> and print one taskset list per GPU.

pack_plan.sh uses it for CPG: a packed GPU case keeps one host core busy, so a cell is only priced fairly when its
cases get the cores the priced box has per GPU. A rented host is shared, so the cores are ranked by idle time over a
2 s sample of /proc/stat and the idlest are taken, one hardware thread per core, each GPU on one socket where possible.
Exit 1 when the container may not use enough physical cores.
"""
import os, subprocess, sys, time

def busy():
    out = {}
    for l in open("/proc/stat"):
        f = l.split()
        if f[0].startswith("cpu") and f[0] != "cpu":
            v = list(map(int, f[1:9])); out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out

def main(cpg, ng):
    allowed = os.sched_getaffinity(0)
    cores = {}   # (socket, core) -> [cpus]
    for l in subprocess.run(["lscpu", "-p=cpu,core,socket"], capture_output=True, text=True).stdout.splitlines():
        if not l.startswith("#"):
            cpu, core, sock = map(int, l.split(",")[:3])
            if cpu in allowed: cores.setdefault((sock, core), []).append(cpu)
    a = busy(); time.sleep(2); b = busy()
    load = lambda c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1])
    # a core is as busy as its busiest thread: a sibling at work halves what the thread we take can do
    ranked = sorted(cores, key=lambda k: (round(max(load(c) for c in cores[k]), 1), k))[:cpg * ng]
    if len(ranked) < cpg * ng: return 1
    ranked.sort()   # by socket, then core: GPU j takes a contiguous run
    for j in range(ng):
        print(",".join(str(min(cores[k])) for k in ranked[j * cpg:(j + 1) * cpg]))
    return 0

if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
