"""Reorder check of two emulator reference files (run_mesh --emuonly --ref): the same case lowered in two lane
orders (or on two partitions) does the same arithmetic with its sums in another order, so the states agree to
rounding and not bitwise. Per state field: relative L2 and largest difference; per history row: the residuals.

    python -m ttfluid.tools.ordercheck BASE.ref LEVER.ref [--tol 1e-2] [--rows 1]

Exit 1 (FAIL) when a state field differs by more than --tol in relative L2, a residual of the first --rows rows by
more than --tol, or a value is not finite. Run it after ONE outer iteration: later rows drift as any re-partition
does (an iteration count that moves by one changes every digit after it). A lane permutation that one field or one
table did not follow gives differences of order one."""
import argparse
import pickle
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("lever")
ap.add_argument("--tol", type=float, default=1e-2)
ap.add_argument("--rows", type=int, default=1)
a = ap.parse_args()
(hb, sb), (hl, sl) = (pickle.load(open(p, "rb")) for p in (a.base, a.lever))
bad, worst = [], 0.0
assert set(sb) == set(sl), sorted(set(sb) ^ set(sl))
for n in sorted(sb):
    x, y = np.asarray(sb[n], np.float64), np.asarray(sl[n], np.float64)
    if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        bad.append(n)
        print("state %-10s shape %s vs %s, finite %s %s" % (n, x.shape, y.shape, np.isfinite(x).all(), np.isfinite(y).all()))
        continue
    rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
    worst = max(worst, rel)
    bad += [n] * (rel > a.tol)
    print("state %-10s n %-9d rel L2 %.3e  max |d| %.3e  max |base| %.3e" % (n, x.size, rel, np.abs(x - y).max(initial=0), np.abs(x).max(initial=0)))
for k in range(min(len(hb), len(hl))):
    out = []
    for n in sorted(hb[k]):
        u, v = float(hb[k][n]), float(hl[k][n])
        if n.startswith("its") or n == "outer":
            out.append("%s %d/%d" % (n, u, v))
            continue
        rel = abs(u - v) / max(abs(u), 1e-30)
        out.append("%s %.2e" % (n, rel))
        if k < a.rows and not rel <= a.tol:
            bad.append("row%d:%s" % (k, n))
    print("row %d%s %s" % (k, " (gated)" if k < a.rows else "", " ".join(out)))
print("%s %s vs %s: worst state rel L2 %.3e, tol %.0e%s" % ("FAIL" if bad else "PASS", a.base.split("/")[-1], a.lever.split("/")[-1], worst, a.tol,
                                                         "; over: " + ",".join(sorted(set(bad))) if bad else ""))
sys.exit(1 if bad else 0)
