"""Cd/Cl window means of a SPUMA run vs OpenFOAM fp64 (CPU v2412), the same way the Tenstorrent runs are graded.

Both sides are OpenFOAM coefficient.dat files from the same potentialFoam start; each window mean carries its batch-means
error (blocks of --block its, cdwindow.py), and the verdict is diff against 2 sigma of the two errors combined.
One count is 1e-4 in Cd or Cl (a drag count), as in cdwindow.py.

    python3 grade.py --run B_DP/postProcessing/forceCoeffsAll/0/coefficient.dat --of of1k.dat of4k.dat [--from 101 --to 3000]
"""
import argparse
import math

from cdwindow import batch, of_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--of", nargs="+", required=True)
    ap.add_argument("--from", dest="lo", type=int, default=101)
    ap.add_argument("--to", dest="hi", type=int, default=3000)
    ap.add_argument("--block", type=int, default=100)
    a = ap.parse_args()
    run, ref = of_series(a.run), of_series(a.of)
    hi = min(a.hi, max(run), max(ref))
    its = [i for i in range(a.lo, hi + 1) if i in run and i in ref]
    print(f"its {its[0]}-{its[-1]} ({len(its)}), block {a.block}")
    for k, name in enumerate(("Cd", "Cl")):
        mr, er, nb = batch([run[i][k] for i in its], a.block)
        mo, eo, _ = batch([ref[i][k] for i in its], a.block)
        d, s2 = (mr - mo) * 1e4, 2e4 * math.hypot(er, eo)
        verdict = "PASS" if abs(d) <= s2 else "FAIL"
        print(f"{name}: run {mr:.5f} +- {2e4*er:.1f} counts, OpenFOAM {mo:.5f} +- {2e4*eo:.1f}; "
              f"diff {d:+.1f} counts, 2 sigma {s2:.1f} ({nb} blocks) {verdict}")


if __name__ == "__main__":
    main()
