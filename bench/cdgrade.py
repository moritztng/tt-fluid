"""Grade the drag of an HPC_motorbike Small run against OpenFOAM v2412 fp64. One tool for device histories and
OpenFOAM/SPUMA coefficient files.

    python3 bench/cdgrade.py --run RUN [RUN ...] [--base RUN ...] [--ref coefficient.dat ...] [--from 201] [--to N]

RUN is `X.hist.json:IT0` (run_mesh --forces history; row k holds Cd_in/Cl_in of the state iteration IT0 + k starts
from, which is OpenFOAM's row for that iteration) or an OpenFOAM coefficient.dat. Several RUNs chain legs of one run.

The case never settles under SIMPLE: Cd swings about 40 counts from iteration to iteration (1 count = 1e-4), and two
runs from the same start are unrelated after about 300 iterations. So a run is graded on the mean over a window, and
the noise of that mean is known in advance from measured runs (docs/accuracy.md, section 3a): the mean of 100 iterations
has a standard deviation of S100 counts, blocks of 100 are close to independent, so a window of n blocks has
S100 / sqrt(n).

    diff  = mean(run, window) - mean(OpenFOAM, its 501 on)       the reference is OpenFOAM's long-run mean
    band  = max(BAND, 2 sigma of diff)                           BAND once the window is long enough to resolve it
    GRADE = PASS if |diff of Cd| <= band else FAIL               exit status 0 / 1

The window starts at 201 because iterations 101-200 still carry a start-up transient that depends on the linear
solvers. With --base the same window of a baseline run is compared too: two correct runs that differ only in the order
of their floating-point sums sit sqrt(2) S100 / sqrt(n) apart (the floor); 1 such pair in 20 is over twice that apart.
"""
import argparse
import json
import math
import sys

BAND = 12.5                       # counts, the accuracy bar on a window long enough to resolve it
S100 = {"Cd": 20.0, "Cl": 18.0}   # counts, sd of a 100-iteration mean (OpenFOAM, SPUMA x18, two device builds)
BLOCK, START, REF_FROM = 100, 201, 501
REF = ["results/openfoam/of1k.dat", "results/openfoam/of4k.dat"]


def series(specs):
    """{iteration: (Cd, Cl)} from history legs and/or coefficient.dat files."""
    s = {}
    for spec in specs:
        if ".json" in spec:
            path, _, it0 = spec.partition(".json")
            for k, r in enumerate(json.load(open(path + ".json"))):
                s[int(it0.lstrip(":") or 0) + k] = (r["Cd_in"], r["Cl_in"])
        else:
            for line in open(spec):
                if line.startswith("# Time"):
                    hdr = line[1:].split()
                elif line.strip() and not line.startswith("#"):
                    v = line.split()
                    s[int(float(v[0]))] = (float(v[hdr.index("Cd")]), float(v[hdr.index("Cl")]))
    return s


def window(s, lo, hi, q):
    """Mean, standard error from block means, and sd of the block means, all of coefficient q over lo..hi."""
    x = [s[i][q] for i in range(lo, hi + 1)]
    bm = [sum(x[b:b + BLOCK]) / BLOCK for b in range(0, len(x) - BLOCK + 1, BLOCK)]
    m, n = sum(x) / len(x), len(bm)
    sd = math.sqrt(sum((b - sum(bm) / n) ** 2 for b in bm) / (n - 1)) if n > 1 else float("nan")
    return m, sd / math.sqrt(n), sd


def grade(run, ref, base=None, lo=START, hi=None):
    hi = max(run) if hi is None else hi
    gaps = [i for i in range(lo, hi + 1) if i not in run or (base is not None and i not in base)]
    if gaps or hi - lo + 1 < BLOCK:
        raise SystemExit(f"window {lo}-{hi}: {len(gaps)} iterations missing (first {gaps[:1]}), need >= {BLOCK}")
    n = (hi - lo + 1) / BLOCK
    out = dict(window=[lo, hi], ref_its=[REF_FROM, max(ref)])
    for q, name in enumerate(("Cd", "Cl")):
        m, se, sd = window(run, lo, hi, q)
        rm, rse, _ = window(ref, REF_FROM, max(ref), q)
        noise = S100[name] / math.sqrt(n)
        sig = math.hypot(noise, rse * 1e4)
        c = dict(run=m, own_se=se * 1e4, block_sd=sd * 1e4, ref=rm, ref_se=rse * 1e4, diff=(m - rm) * 1e4,
                 noise=noise, band=max(BAND, 2 * sig), detects=max(BAND, 2 * sig) + 1.28 * sig,
                 steady=not sd * 1e4 > S100[name] * (1 + 2 / math.sqrt(2 * max(n - 1, 1))))
        if all(i in ref for i in (lo, hi)):
            c["diff_same_its"] = (m - window(ref, lo, hi, q)[0]) * 1e4
        if base is not None:
            c["shift"], c["floor"] = (m - window(base, lo, hi, q)[0]) * 1e4, math.sqrt(2) * noise
        c["ok"] = abs(c["diff"]) <= c["band"]   # a NaN in the window fails
        out[name] = c
    out["grade"] = "PASS" if out["Cd"]["ok"] else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--base", nargs="+")
    ap.add_argument("--ref", nargs="+", default=REF)
    ap.add_argument("--from", dest="lo", type=int, default=START)
    ap.add_argument("--to", dest="hi", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    g = grade(series(a.run), series(a.ref), series(a.base) if a.base else None, a.lo, a.hi)
    lo, hi = g["window"]
    print(f"window {lo}-{hi} ({hi - lo + 1} its), reference OpenFOAM its {g['ref_its'][0]}-{g['ref_its'][1]}")
    for name in ("Cd", "Cl"):
        c = g[name]
        same = f", same its {c['diff_same_its']:+.1f}" if "diff_same_its" in c else ""
        print(f"{name} {c['run']:.5f} +- {c['noise']:.1f} counts (own batch error {c['own_se']:.1f}) | OpenFOAM "
              f"{c['ref']:.5f} +- {c['ref_se']:.1f} | diff {c['diff']:+.1f} counts{same} | band {c['band']:.1f} "
              f"{'inside' if c['ok'] else 'OUTSIDE'}")
        if not c["steady"]:
            print(f"   block means spread {c['block_sd']:.1f} counts against {S100[name]:.0f} in the runs the band was "
                  "set from: this run is less steady")
        if "shift" in c:
            print(f"   vs base {c['shift']:+.1f} counts, floor {c['floor']:.1f} (1 sigma): "
                  f"{'inside' if abs(c['shift']) <= 2 * c['floor'] else 'OUTSIDE'} 2 sigma")
    c = g["Cd"]
    print(f"GRADE: {g['grade']}  Cd {c['diff']:+.1f} counts against a band of {c['band']:.1f} on {lo}-{hi}; a bias of "
          f"{c['detects']:.0f} counts fails 9 times in 10")
    if a.json:
        json.dump(g, open(a.json, "w"), indent=1)
    sys.exit(g["grade"] != "PASS")


if __name__ == "__main__":
    main()
