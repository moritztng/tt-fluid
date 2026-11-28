"""flx-kernel: leg lines of a chain status file with the in-run sysfs AICLK. absum.py STATUS PREFIX
In-run = the samples from the first to the last one at or above 1300 MHz (the node idles at 800 before the open and
after the close); a sample under 1300 inside that span voids the leg."""
import json, re, sys, os
st, pre = sys.argv[1], sys.argv[2]
L = os.path.dirname(st)
rows = {}
for ln in open(st, errors="replace"):
    m = re.search(r" (%s_\w+) rc=(\d+) img=(\S+) blocks=(\d) host=(\S+) chip=(\S+) sha=(\S+) aiclk=\S+ (?:samples=\d+ )?ms_device=(\S*) digest=(\w*)" % re.escape(pre), ln)
    if not m:
        continue
    n, rc, img, blk, host, chip, sha, ms, dig = m.groups()
    v = []
    try:
        for a in open(os.path.join(L, n + ".aiclk")):
            v += [int(x.split(":")[1]) for x in a.split()[2:] if x.startswith("n")]
    except OSError:
        pass
    hi = [i for i, x in enumerate(v) if x >= 1300]
    run = v[hi[0]:hi[-1] + 1] if hi else []
    rows[n] = dict(rc=int(rc), img=img, blocks=int(blk), host=host, chip=chip, sha=sha, ms=float(ms) if ms else None, dig=dig,
                   aiclk="%d..%d" % (min(run), max(run)) if run else "none", ns=len(run), void=bool(run) and min(run) < 1300)
for n, r in rows.items():
    print("%-8s rc=%d blocks=%d host=%s chip=%s sha=%s aiclk=%s (%d sysfs samples in the run%s) ms_device=%s digest=%s"
          % (n, r["rc"], r["blocks"], r["host"], r["chip"], r["sha"], r["aiclk"], r["ns"], ", VOID" if r["void"] else "", "%.2f" % r["ms"] if r["ms"] else "-", r["dig"][:16]))
kept = lambda n: re.search(r"_[AB][1-9]$", n)   # not the first runs (A0, B0, A0c, B0c), not the --prof legs (Ap, Bp)
A = [r["ms"] for n, r in rows.items() if r["blocks"] == 0 and r["rc"] == 0 and r["ms"] and kept(n) and not r["void"]]
B = [r["ms"] for n, r in rows.items() if r["blocks"] == 1 and r["rc"] == 0 and r["ms"] and kept(n) and not r["void"]]
dA = {r["dig"] for n, r in rows.items() if r["blocks"] == 0 and r["rc"] == 0}
dB = {r["dig"] for n, r in rows.items() if r["blocks"] == 1 and r["rc"] == 0}
dX = {r["dig"] for n, r in rows.items() if r["blocks"] == 2}
if A and B:
    ma, mb = sum(A) / len(A), sum(B) / len(B)
    print("A mean %.2f ms (n=%d, %s) | B mean %.2f ms (n=%d, %s) | A/B %.3fx | digests: B == A %s, X == A %s"
          % (ma, len(A), " ".join("%.2f" % x for x in A), mb, len(B), " ".join("%.2f" % x for x in B), ma / mb,
             dA == dB and len(dA) == 1, bool(dX) and dX <= dA))
