"""Grade the drag of an HPC_motorbike Small run against OpenFOAM v2412 fp64. One tool for device histories and
OpenFOAM/SPUMA coefficient files.

    python3 bench/cdgrade.py --run RUN [RUN ...] [--base RUN ...] [--pool RUN ...]... [--ref coefficient.dat ...]
        [--from 201] [--to N] [--means]

RUN is `X_state.npz:IT0` or `X.hist.json:IT0` (the history run_mesh --forces saves with the final state; row k holds
Cd_in/Cl_in of the state iteration IT0 + k starts from, which is OpenFOAM's row for that iteration) or an OpenFOAM
coefficient.dat. Several RUNs chain legs of one run.

The case never settles under SIMPLE: Cd swings about 40 counts from iteration to iteration (1 count = 1e-4), and two
runs from the same start are unrelated after about 300 iterations. So a run is graded on the mean over a window, and
the noise of that mean is known in advance from measured runs (docs/accuracy.md, section 3): the mean of 100 iterations
has a standard deviation of S100 counts and blocks of 100 are close to independent, so a window of n blocks has
S100 / sqrt(n). That holds up to about 600 iterations. Beyond it the mean of one device run wanders: windows of
1,100 to 1,900 iterations spread 10 counts between and within runs, twice what independent blocks give. So one run
counts as ONE_RUN blocks at most, however long its window, and the noise comes down only with more runs (--pool).

    diff  = mean(run, window) - mean(OpenFOAM, its 501 on)       the reference is OpenFOAM's long-run mean
    band  = max(BAND, 2 sigma of diff)                           BAND once enough runs are pooled to resolve it
    GRADE = PASS if |diff of Cd| <= band else FAIL               exit status 0 / 1

The window starts at 201 because iterations 101-200 still carry a start-up transient that depends on the linear
solvers. With --base the same window of a baseline run is compared too: two correct runs that differ only in the order
of their floating-point sums sit sqrt(2) S100 / sqrt(n) apart (the floor); 1 such pair in 20 is over twice that apart.

A change that only reorders the same arithmetic has a much sharper check, also printed with --base when both runs
start from the same state: such runs still track each other through iteration 10 (TRACK), long before the window.
After that one linear solve taking one iteration more or fewer separates them for good. A run too short for a window
(11 iterations are enough) gets this check alone, and the exit status is its result.

Runs that track each other are draws of one arithmetic, so their windows can be pooled into one grade: each --pool
adds such a run (the tool refuses one that does not track --run), and the pooled window counts every run's blocks
except the first block of each added run, which still follows the first run. The pooled sigma is the larger of that
model and the spread of the run means themselves. RECORD_RUNS runs or more make a grade of record; fewer are a screen.

The band and the noise model are measured on this one case. For any other case --means prints the window means of the
run and of --ref over the same iterations with their batch errors, and no grade.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

BAND = 12.5                       # counts, the accuracy bar once enough runs are pooled to resolve it
ONE_RUN, RECORD_RUNS = 6, 4       # blocks one run counts for at most; runs a grade of record pools
S100 = {"Cd": 20.0, "Cl": 18.0}   # counts, sd of a 100-iteration mean (OpenFOAM, SPUMA x18, two device builds)
TRACK_ITS, TRACK = 10, 1.0        # reordered runs stay within TRACK counts of Cd through iteration TRACK_ITS
BLOCK, START, REF_FROM = 100, 201, 501
REF = [str(Path(__file__).resolve().parent.parent / "results/openfoam" / f) for f in ("of1k.dat", "of4k.dat")]


def series(specs):
    """{iteration: (Cd, Cl)} from history legs and/or coefficient.dat files."""
    s = {}
    for spec in specs:
        leg = re.fullmatch(r"(.+\.(json|npz))(:(\d+))?", spec)
        if leg and leg[2] == "npz":
            import numpy as np
            rows = json.loads(str(np.load(leg[1])["hist"]))
        elif leg:
            rows = json.load(open(leg[1]))
        if leg:
            for k, r in enumerate(rows):
                s[int(leg[4] or 0) + k] = (r["Cd_in"], r["Cl_in"])
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


def tracks(run, base):
    """Largest |Cd difference| in counts through iteration TRACK_ITS, or None unless both runs hold those iterations."""
    its = range(1, TRACK_ITS + 1)
    return max(abs(run[i][0] - base[i][0]) for i in its) * 1e4 if all(i in run and i in base for i in its) else None


def grade(run, ref, base=None, lo=START, hi=None):
    hi = max(run) if hi is None else hi
    gaps = [i for i in range(lo, hi + 1) if i not in run or (base is not None and i not in base)]
    if gaps or hi - lo + 1 < BLOCK:
        raise SystemExit(f"no window: the run ends at iteration {hi} and a window is {lo}-{lo + BLOCK - 1} or longer"
                         if hi - lo + 1 < BLOCK else f"window {lo}-{hi}: {len(gaps)} iterations missing, first {gaps[0]}")
    blocks = (hi - lo + 1) / BLOCK
    n = min(blocks, ONE_RUN)
    out = dict(window=[lo, hi], ref_its=[REF_FROM, max(ref)], blocks=n)
    for q, name in enumerate(("Cd", "Cl")):
        m, se, sd = window(run, lo, hi, q)
        rm, rse, _ = window(ref, REF_FROM, max(ref), q)
        noise = S100[name] / math.sqrt(n)
        sig = math.hypot(noise, rse * 1e4)
        c = dict(run=m, own_se=se * 1e4, block_sd=sd * 1e4, ref=rm, ref_se=rse * 1e4, diff=(m - rm) * 1e4,
                 noise=noise, band=max(BAND, 2 * sig), detects=max(BAND, 2 * sig) + 1.28 * sig,
                 steady=not sd * 1e4 > S100[name] * (1 + 2 / math.sqrt(2 * max(blocks - 1, 1))))
        if all(i in ref for i in (lo, hi)):
            c["diff_same_its"] = (m - window(ref, lo, hi, q)[0]) * 1e4
        if base is not None:
            c["shift"], c["floor"] = (m - window(base, lo, hi, q)[0]) * 1e4, math.sqrt(2) * noise
        c["ok"] = abs(c["diff"]) <= c["band"]   # a NaN in the window fails
        out[name] = c
    out["grade"] = "PASS" if out["Cd"]["ok"] else "FAIL"
    if base is not None and (t := tracks(run, base)) is not None:
        out["early"], out["reordered"] = t, t <= TRACK   # a NaN does not track
    return out


def pooled(grades):
    """One grade from the grades of several runs of the same arithmetic: the mean over all their window iterations."""
    its = [g["window"][1] - g["window"][0] + 1 for g in grades]
    r, n = len(grades), sum(g["blocks"] for g in grades) - (len(grades) - 1)
    out = dict(runs=r, its=sum(its), record=r >= RECORD_RUNS)
    for name in ("Cd", "Cl"):
        diff = sum(g[name]["diff"] * k for g, k in zip(grades, its)) / sum(its)
        model = S100[name] / math.sqrt(n)
        spread = math.sqrt(sum((k / sum(its) * (g[name]["diff"] - diff)) ** 2 for g, k in zip(grades, its)) * r / (r - 1))
        sig = math.hypot(max(model, spread), grades[0][name]["ref_se"])
        out[name] = dict(diff=diff, sigma=sig, model=model, spread=spread, band=max(BAND, 2 * sig),
                         detects=max(BAND, 2 * sig) + 1.28 * sig, ok=abs(diff) <= max(BAND, 2 * sig))
    out["grade"] = "PASS" if out["Cd"]["ok"] else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--base", nargs="+")
    ap.add_argument("--pool", nargs="+", action="append", default=[], help="another run of the same arithmetic")
    ap.add_argument("--ref", nargs="+", default=REF)
    ap.add_argument("--from", dest="lo", type=int, default=START)
    ap.add_argument("--to", dest="hi", type=int)
    ap.add_argument("--means", action="store_true", help="another case: window means against --ref, no grade")
    ap.add_argument("--json")
    a = ap.parse_args()
    run, base = series(a.run), series(a.base) if a.base else None
    if a.means:
        ref, lo, hi = series(a.ref), a.lo, max(run) if a.hi is None else a.hi
        for q, name in enumerate(("Cd", "Cl")):
            (m, se, _), (rm, rse, _) = window(run, lo, hi, q), window(ref, lo, hi, q)
            print(f"{name} its {lo}-{hi}: run {m:.5f} +- {se * 1e4:.1f} counts | reference {rm:.5f} +- {rse * 1e4:.1f} | "
                  f"diff {(m - rm) * 1e4:+.1f} +- {math.hypot(se, rse) * 1e4:.1f} counts (1 sigma, batch means)")
        return

    def track(t):
        print(f"its 1-{TRACK_ITS} vs base: Cd within {t:.2f} counts | the same arithmetic in another order stays within "
              f"{TRACK:g}: " + ("it tracks the base" if t <= TRACK else "this is NOT a reorder of the base"))

    if base is not None and a.hi is None and max(run) < a.lo + BLOCK - 1 and (t := tracks(run, base)) is not None:
        track(t)
        sys.exit(not t <= TRACK)
    ref = series(a.ref)
    g = grade(run, ref, base, a.lo, a.hi)
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
    if "early" in g:
        track(g["early"])
    c = g["Cd"]
    print(f"GRADE: {g['grade']}  Cd {c['diff']:+.1f} counts against a band of {c['band']:.1f} on {lo}-{hi}; a bias of "
          f"{c['detects']:.0f} counts fails 9 times in 10")
    if a.pool:
        grades = [g]
        for spec in a.pool:
            r = series(spec)
            if (t := tracks(r, run)) is None or not t <= TRACK:
                raise SystemExit(f"not pooled: {spec[0]} does not track --run through iteration {TRACK_ITS} "
                                 f"({'those iterations are missing' if t is None else f'{t:.2f} counts'})")
            grades.append(grade(r, ref, None, a.lo, a.hi))
            w, c = grades[-1]["window"], grades[-1]["Cd"]
            print(f"pooled with {spec[0]}: window {w[0]}-{w[1]}, Cd {c['diff']:+.1f} counts, tracks to {t:.2f}")
        g["pooled"] = q = pooled(grades)
        g["grade"], c = q["grade"], q["Cd"]
        print(f"POOLED GRADE: {q['grade']}  Cd {c['diff']:+.1f} +- {c['sigma']:.1f} counts (Cl {q['Cl']['diff']:+.1f}) "
              f"against a band of {c['band']:.1f} on {q['its']} iterations of {q['runs']} runs; a bias of "
              f"{c['detects']:.0f} counts fails 9 times in 10 | sigma from the run means {c['spread']:.1f}, from the "
              f"model {c['model']:.1f}"
              + ("" if q["record"] else f" | fewer than {RECORD_RUNS} runs: a screen, not a grade of record"))
    if a.json:
        json.dump(g, open(a.json, "w"), indent=1)
    sys.exit(g["grade"] != "PASS")


if __name__ == "__main__":
    main()
