"""flx-kernel: which planned bodies does blockgen turn down, and why. rejects.py MCACHE.pkl[.zst]  (CPU only)"""
import collections, io, pickle, sys
import numpy as np
from ttfluid.resident import blklower, blockgen

def load(p):
    if p.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            return pickle.load(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True))
    with open(p, "rb") as f:
        return pickle.load(f)

saved, prog, m = load(sys.argv[1])
why = collections.Counter()
shown = set()
def acc(sp):
    if blockgen.accept(sp):
        return True
    try:
        cfg, row = blockgen._plain(sp)
        r = "row of %d instructions over 32" % len(row) if len(row) > 32 else "replay differs: %s" % (blockgen.check(sp),)
    except blockgen.Reject as e:
        r = "Reject: %s" % (e,)
    except Exception as e:
        r = "%s: %s" % (type(e).__name__, e)
    why[r[:60]] += 1
    k = tuple(o[0] for o in sp["ops"])
    if k not in shown and len(shown) < 12:
        shown.add(k)
        print("REJECTED", r, "| ops", sp["ops"], "| ins", sp["ins"], "| outs", sp["outs"], "| const", sp["const"], flush=True)
    return False

info = collections.Counter()
nb = 0
for ins in m.mir:
    if ins[0] not in ("blk", "seg"):
        continue
    for b in ([ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]):
        nb += 1
        blklower.lower_block(b[1], b[2], m.nslot, accept=acc, info=info)
print("blocks", nb, dict(info))
print("why", dict(why))
