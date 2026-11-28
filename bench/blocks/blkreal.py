"""flx-kernel: the plain bodies as generated, priced per tile-it against the stock op words (CPU only).
blkreal.py MCACHE.pkl[.zst] [REC]   REC = cycles per pushed config / record word when a body is recorded every tile
Static count as tools/blkprice.py: a block weighs its passes (tiles when it loops), per tile-it = / T."""
import collections, pickle, sys
from ttfluid.resident import blklower, blockgen

def load(p):
    if p.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            return pickle.load(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True))
    with open(p, "rb") as f:
        return pickle.load(f)

saved, prog, m = load(sys.argv[1])
REC = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
T = m.T
A = collections.Counter(); hist = collections.Counter(); kinds = collections.Counter(); left = collections.Counter()
for ins in m.mir:
    if ins[0] not in ("blk", "seg"):
        continue
    blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]
    bt = m.tiles(blks)
    for b in blks:
        vec = any(u[0] == "ldr" or (u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__" and m.is_vec(u[2])) for u in b[1]) or any(m.is_vec(x) for _, x in b[2])
        w = (bt if (ins[0] == "seg" or vec) else 1) / T
        items, _ = blklower.lower_block(b[1], b[2], m.nslot, accept=blockgen.accept)
        A["stock before"] += w * blockgen.stock_cost(b[1])
        A["stock left"] += w * blockgen.stock_cost([u for u in items if u[0] != "body"])
        for u in items:
            if u[0] in ("op", "un"):
                left[u[1]] += w
            elif u[0] in ("fill", "cpd", "fold"):
                left[u[0]] += w
            if u[0] != "body":
                continue
            cfg, row = blockgen.program(u[1], sym=True)
            L = len(row)
            A["bodies"] += w; A["ops"] += w * len(u[1]["ops"]); A["instr"] += w * L; A["cfg"] += w * len(cfg)
            A["words"] += w * (1 + len(cfg) + L * (1 if L >= 4 else 2))
            for x in row:
                kinds[x[0]] += w
            hist[L] += w
            A["replay"] += w * 32 * L
            A["fixed"] += w * 17
            A["record"] += w * REC * (len(cfg) + L * (1 if L >= 4 else 2) + 1)
print("T %d nslot %d | per tile-it: bodies %.1f, ops in bodies %.1f, row instructions %.1f (%.2f per body, %.2f per op), config words %.1f, inline words %.1f"
      % (T, m.nslot, A["bodies"], A["ops"], A["instr"], A["instr"] / A["bodies"], A["instr"] / A["ops"], A["cfg"], A["words"]))
print("instructions by kind per tile-it:", {k: round(v, 1) for k, v in kinds.most_common()})
print("row length histogram (body tiles per tile-it):", {k: round(v, 1) for k, v in sorted(hist.items())})
print("stock ops left per tile-it:", {k: round(v, 1) for k, v in left.most_common()})
M = 1e-6
body = A["fixed"] + A["replay"]
print("M cycles per tile-it: stock op words today %.3f | with plain bodies: bodies %.3f (fixed 17 a body %.3f + rows %.3f) + stock left %.3f = %.3f (%.2fx)"
      % (A["stock before"] * M, body * M, A["fixed"] * M, A["replay"] * M, A["stock left"] * M, (body + A["stock left"]) * M, (body + A["stock left"]) / A["stock before"]))
print("  + record and config every body-tile at %.1f cycles a pushed word: %.3f -> total %.3f (%.2fx)"
      % (REC, A["record"] * M, (body + A["stock left"] + A["record"]) * M, (body + A["stock left"] + A["record"]) / A["stock before"]))
