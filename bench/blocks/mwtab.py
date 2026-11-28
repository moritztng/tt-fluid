"""flx-kernel mwtab.py: who MATH waits for, from run_mesh --prof 2 legs (census.mwait, vm_compute.cpp CFD_PROFW).

    python3 mwtab.py W1.jsonl [W2.jsonl ...] [--prof P.jsonl] [--plain B.jsonl] [--its 20] [--T 6]

Per leg: MATH's cycles a core (mean over cores) spent waiting at its loads and acquires, by cause, as M cycles per
tile-it (cycles / (its x T)) and as a share of MATH's whole run. The causes are the states the waited-for threads
publish at each poll of MATH (stdlib only; the names are census.MW_U / MW_N / MW_P):
  a load waits for the unpacker; UNPACK is  "at the load" (interpreting or unpacking: the L1 handshake itself),
  or waits for a "reader page" (the next ring page of NCRISC), a token or a code page; the reader then is in its
  "DRAM barrier" (a DRAM page in flight), its "patch barrier" (the entries a gather missed, read from DRAM),
  GATHER / scatter (the gather's L1 work), "halo wait", "slot wait", or waits for a token itself (WAITC, WAITB).
  an acquire waits for the packer; PACK is "at work" (packing), waits for a "writer page" (store back-pressure) ...
Check lines: the buckets against MATH's own op slots of the same leg (LD + LDS + LDR+STW and ACQ), and the legs'
ms_device against a plain --prof leg (--prof) and a plain leg (--plain): what the instrument itself costs.
"""
import json, sys


def arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


def last(path):
    r = None
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                r = json.loads(ln)
            except ValueError:
                pass
    return r


def ms(r):
    v = r.get("ms_device")
    return float(v[0] if isinstance(v, list) else v)


def cause(u, n):
    if u == "at the load":
        return "L1 handshake itself (UNPACK at the load)"
    if u == "reader page":
        return {"DRAM barrier": "DRAM page in flight (STREAM's read barrier)",
                "patch barrier": "gather miss (patch entries read from DRAM)",
                "GATHER": "gather, L1 work", "scatter": "gather, L1 work", "halo wait": "gather waits for the halo",
                "STREAM": "stream, the reader's own work", "slot wait": "stream waits for a free ring slot",
                "WAITC": "reader waits for compute's token", "WAITB": "reader waits for BRISC's token",
                "HALO": "reader's HALO op"}.get(n, "reader page, other (%s)" % n)
    return "UNPACK waits for a %s" % u


def table(path, its, T):
    r = last(path)
    c = r["census"]
    w, m = c["mwait"], c["math"]
    tot, ops = m["total"], m["ops"]
    per = lambda x: x / (its * T) / 1e6
    rows = {}
    for k, v in w["load"].items():
        if k[:1] == "u" and k[1:2].isdigit():   # legs of 132b2c2: words 952 and 954 (the acquire's cross-checks) were also read as load buckets "u7 | n14/15"
            continue
        u, n = k.split(" | ")
        rows[cause(u, n)] = rows.get(cause(u, n), 0) + v
    out = ["%s  ms_device %.2f  MATH %.4g cycles a core = %.1f M per tile-it (its %d, T %d)" % (path, ms(r), tot, per(tot), its, T),
           "  loads (n %.0f a core): %s" % (w["n"]["load"], " ".join("%s x %.0f" % (k, ops[k]["n"]) for k in ("LD", "LDS", "LDR+STW") if k in ops))]
    line = lambda nm, v: "    %-48s %8.2f M  %5.1f %%" % (nm, per(v), 100 * v / tot)
    for nm, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        out.append(line(nm, v))
    lsum = sum(rows.values())
    out.append(line("= waited for the unpacker, all causes", lsum))
    out.append(line("last poll of each load + the get and clears", w["tail"]["load"] + w["load_after"] + w["tail"]["load_after"]))
    out.append("  acquires (n %.0f a core):" % w["n"]["acq"])
    for nm, v in sorted(w["acq"].items(), key=lambda kv: -kv[1]):
        out.append(line("PACK " + nm + (" (store back-pressure)" if nm == "writer page" else ""), v))
    asum = sum(w["acq"].values())
    out.append(line("= waited for the packer, all causes", asum))
    out.append(line("last poll of each acquire + the drain after it", w["tail"].get("acq", 0) + w.get("acq_after", 0) + w["tail"].get("acq_after", 0)))
    out.append(line("MATH's own queued work, drained at loads and acquires", w["own"] + w["tail"]["own"]))
    slots = sum(ops[k]["cyc"] for k in ("LD", "LDS", "LDR+STW", "ACQ") if k in ops)
    booked = lsum + asum + w["own"] + w["load_after"] + w.get("acq_after", 0) + sum(w["tail"].values())
    out.append("  check: booked %.2f M of the LD + LDS + LDR+STW + ACQ slots' %.2f M per tile-it (%.1f %%; the rest is interpreting)"
               % (per(booked), per(slots), 100 * booked / slots))
    return r, out, dict(rows, **{"PACK " + k: v for k, v in w["acq"].items()}), tot


if __name__ == "__main__":
    its, T = arg("--its", 20, int), arg("--T", 6, int)
    prof, plain = arg("--prof"), arg("--plain")
    legs = [table(p, its, T) for p in sys.argv[1:]]
    for _, out, _, _ in legs:
        print("\n".join(out))
    if len(legs) > 1:   # leg to leg: every cause's spread as a share of MATH's run
        keys = sorted(set().union(*[set(l[2]) for l in legs]))
        print("leg to leg, share of MATH's run by cause: " + "; ".join(
            "%s %s" % (k, "/".join("%.1f" % (100 * l[2].get(k, 0) / l[3]) for l in legs)) for k in keys))
    for nm, p in (("--prof", prof), ("plain", plain)):
        if p:
            b = ms(last(p))
            print("instrument cost against the %s leg %s: ms_device %s vs %.2f = %s"
                  % (nm, p, ", ".join("%.2f" % ms(l[0]) for l in legs), b, ", ".join("%.3fx" % (ms(l[0]) / b) for l in legs)))
