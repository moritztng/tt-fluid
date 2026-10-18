"""Windowed Cd/Cl gate: device per-iteration coefficients (run_mesh --forces history) vs OpenFOAM coefficient.dat.

On a case whose Cd oscillates from iteration to iteration (HPC_motorbike Small: sd ~42 counts over its 101-800 in
OpenFOAM's own run), a single-iteration comparison says nothing. Both sides are averaged over a window and the error of
each mean comes from batch means (blocks of --block its), so the verdict reads diff +- 2 sigma against the gate.

    python3 cdwindow.py --dev F8.hist.json:700 F9.hist.json:800 ... --of coefficient.dat [coefficient.dat ...]
        [--from 701] [--to 3000] [--of-from 501] [--block 100] [--gate 3]

A history file's row k (0-based) carries Cd_in/Cl_in, the coefficients of the state the outer iteration starts from,
i.e. OpenFOAM's row for iteration IT0 + k, where IT0 is the number after the colon.
"""
import argparse
import json

import numpy as np


def dev_series(specs):
    s = {}
    for spec in specs:
        path, it0 = spec.rsplit(":", 1)
        for k, r in enumerate(json.load(open(path))):
            s[int(it0) + k] = (r["Cd_in"], r["Cl_in"])
    return s


def of_series(paths):
    s = {}
    for path in paths:
        hdr = None
        for line in open(path):
            if line.startswith("# Time"):
                hdr = line[1:].split()
            elif not line.startswith("#") and line.strip():
                v = line.split()
                s[int(float(v[0]))] = (float(v[hdr.index("Cd")]), float(v[hdr.index("Cl")]))
    return s


def batch(x, block):
    """Mean and its standard error from non-overlapping block means."""
    nb = len(x) // block
    if nb < 2:
        return float(np.mean(x)), float("nan"), nb
    bm = np.asarray(x[: nb * block]).reshape(nb, block).mean(1)
    return float(np.mean(x)), float(bm.std(ddof=1) / np.sqrt(nb)), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--of", nargs="+", required=True)
    ap.add_argument("--from", dest="lo", type=int, default=None)
    ap.add_argument("--to", dest="hi", type=int, default=None)
    ap.add_argument("--of-from", type=int, default=501)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--gate", type=float, default=3.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d, o = dev_series(a.dev), of_series(a.of)
    lo = a.lo if a.lo is not None else min(d)
    hi = a.hi if a.hi is not None else max(d)
    its = [i for i in range(lo, hi + 1) if i in d]
    gaps = (hi - lo + 1) - len(its)
    oits = [i for i in sorted(o) if i >= a.of_from]
    out = dict(dev_its=[its[0], its[-1], len(its)], dev_gaps=gaps, of_its=[oits[0], oits[-1], len(oits)], block=a.block)
    for q, name in enumerate(("Cd", "Cl")):
        dm, de, dn = batch([d[i][q] for i in its], a.block)
        sm, se, _ = batch([o[i][q] for i in its if i in o], a.block)
        om, oe, on = batch([o[i][q] for i in oits], a.block)
        diff, sig = (dm - om) * 1e4, np.hypot(de, oe) * 1e4
        out[name] = dict(dev=dm, dev_se=de * 1e4, of_same=sm, of_same_se=se * 1e4, of_long=om, of_long_se=oe * 1e4,
                         diff_counts=diff, sigma_counts=sig, diff_same_counts=(dm - sm) * 1e4)
        print("%s dev its %d-%d mean %.5f +- %.1f counts (%d blocks) | OF same its %.5f +- %.1f | OF its %d-%d %.5f +- %.1f"
              " (%d blocks) | dev - OF long %+.1f +- %.1f counts (1 sigma), same its %+.1f | OF sd over the window %.0f"
              % (name, its[0], its[-1], dm, de * 1e4, dn, sm, se * 1e4, oits[0], oits[-1], om, oe * 1e4, on, diff, sig,
                 (dm - sm) * 1e4, np.std([o[i][q] for i in its if i in o]) * 1e4))
    c = out["Cd"]
    verdict = ("PASS" if abs(c["diff_counts"]) + 2 * c["sigma_counts"] <= a.gate else
               "FAIL" if abs(c["diff_counts"]) - 2 * c["sigma_counts"] > a.gate else "UNRESOLVED")
    out["verdict"] = verdict
    print("Cd gate %.1f counts: %s (|diff| %.1f, 2 sigma %.1f)%s" % (a.gate, verdict, abs(c["diff_counts"]),
          2 * c["sigma_counts"], ", %d device its missing" % gaps if gaps else ""))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
