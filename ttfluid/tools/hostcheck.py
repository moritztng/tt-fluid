"""An emulator reference (run_mesh --emuonly --ref) against the host solver run in the same ELL slot order, fp32,
after the reference's number of SIMPLE iterations (use one).

    python -m ttfluid.tools.hostcheck CASE REF face|dir [TOL]

This is the check of a lowering that changes the slot order (--order box: slots by face direction). ordercheck
against a cell-id reference does not apply to it: on a uniform start limitedLinear picks upwind or linear by the sign
of a zero gradient's rounding noise wherever the field has no jump across a face (pitzDaily: 3,049 of 24,170 faces
for k), the face value is the same and the matrix split is not, and one iteration later k differs by 2 % in fp64
on the host alone. Exit 1 when a field's relative L2 is over TOL (1e-2)."""
import pickle
import sys
import numpy as np
from ttfluid.solver.simple import SimpleFoam
hist, st = pickle.load(open(sys.argv[2], "rb"))
s = SimpleFoam(sys.argv[1], backend="ell", dtype=np.float32, log=lambda *x: None, ell_order=sys.argv[3])
for _ in range(len(hist)):
    s.iterate()
tol, worst = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2, 0.0
for n, f in zip(("U", "p", "k", "omega", "nut"), s.fields()):
    x, y = np.asarray(f.internal, np.float64).ravel(), np.asarray(st[n], np.float64).ravel()
    rel = np.linalg.norm(x - y) / np.linalg.norm(x)
    worst = max(worst, rel)
    print("%-6s rel L2 %.3e max |d| %.3e of %.3e" % (n, rel, np.abs(x - y).max(), np.abs(x).max()))
print("%s %s against the host solver with slots by %s, %d it: worst %.3e (tol %.0e)" % ("PASS" if worst <= tol else "FAIL", sys.argv[2].split("/")[-1], sys.argv[3], len(hist), worst, tol))
sys.exit(worst > tol)
