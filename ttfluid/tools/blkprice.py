"""flx-kernel: price generated SFPU blocks on a lowered machine before building them (LEVERS 13, DECISIONS D22).

blkprice.py MCACHE.pkl[.zst] [--json OUT]   Walks the MIR like the encoder, plans every block with
resident/blockgen.plan under each policy and sums cycles per region with the measured unit costs: the ops as stock
tile calls (today), as bodies in today's uop order (no hoisting), and with loads hoisted into free Dest slots for
4 and 8 slots. Per tile-it = per core and pass of the region / T. Loads, stores and DRAM pages are not in these sums."""
import argparse
import io
import json
from collections import Counter

from ttfluid.resident import blockgen as bg
from ttfluid.resident.census import region_of


def load(path):
    import cloudpickle
    f = open(path, "rb")
    if path.endswith(".zst"):
        import zstandard
        f = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True, closefd=True), 1 << 24)
    with f:
        return cloudpickle.load(f)


POL = {"order4": dict(slots=4, hoist=False), "hoist4": dict(slots=4, hoist=True), "hoist8": dict(slots=8, hoist=True)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json")
    a = ap.parse_args()
    LOWER, prog, m = load(a.path)
    mir, reg = m.mir, region_of(m.mir)
    R = {}
    for j, ins in enumerate(mir):
        if ins[0] not in ("blk", "seg"):
            continue
        r = R.setdefault(reg[j], dict(stock=0.0, kinds=Counter(), est=0.0, blocks=0, tiles=0, shape=Counter(),
                                      **{p: dict(cyc=0.0, bodies=0, instr=0, fused=0, left=0) for p in POL}))
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]
        bt = m.tiles(blks)
        for b in blks:
            vec = any(u[0] == "ldr" or (u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__" and m.is_vec(u[2]))
                      for u in b[1]) or any(m.is_vec(x) for _, x in b[2])
            passes = bt if (ins[0] == "seg" or vec) else 1
            r["blocks"] += 1
            r["tiles"] += passes
            for u in b[1]:
                k = u[1] if u[0] in ("op", "un") else u[0]
                r["kinds"][k] += passes
                if u[0] in ("op", "un", "fill", "cpd", "fold") and k not in bg.MEASURED:
                    r["est"] += bg.STOCK[k] * passes
            r["stock"] += bg.stock_cost(b[1]) * passes
            for p, kw in POL.items():
                pl, ev, uses = bg.plan(b[1], b[2], **kw)
                c, _, nb, ni, nin, nst = bg.cost(pl)
                d = r[p]
                d["cyc"] += c * passes
                d["bodies"] += nb * passes
                d["instr"] += ni * passes
                d["fused"] += nin * passes
                d["left"] += nst * passes
                if p == "hoist4":
                    for k, q in pl:
                        if k == "body":
                            r["shape"][(len(q["ins"]), len(q["ops"]), len(q["outs"]))] += passes
    T = m.T
    print(a.path.split("/")[-1], "T", T, "nslot", m.nslot)
    for name, r in sorted(R.items()):
        ar = sum(v for k, v in r["kinds"].items() if k not in ("ld", "lds", "ldr", "rvf"))
        print(f"{name}: blocks {r['blocks']}, block tiles {r['tiles']}, arithmetic uop tiles {ar} ({ar / T:.0f} per tile-it)")
        print(f"  stock      {r['stock'] / T / 1e6:7.3f} M cycles per tile-it ({100 * r['est'] / max(r['stock'], 1):.0f} % of it on EST unit costs)")
        for p in POL:
            d = r[p]
            print(f"  {p:10s} {d['cyc'] / T / 1e6:7.3f} M  = {d['cyc'] / max(r['stock'], 1):.2f}x stock; bodies {d['bodies'] / T:.0f} per tile-it, "
                  f"{d['instr'] / max(d['bodies'], 1):.1f} instr and {d['fused'] / max(d['bodies'], 1):.2f} ops per body, "
                  f"{(d['cyc'] - 0) / max(d['fused'] + d['left'], 1):.0f} cyc per op, {d['left'] / T:.0f} stock ops left per tile-it")
        print("  kinds per tile-it:", {k: round(v / T) for k, v in r["kinds"].most_common()})
        print("  hoist4 bodies (operands, ops, results): tiles", [(k, v) for k, v in r["shape"].most_common(12)])
    if a.json:
        json.dump({n: dict(stock=r["stock"], est=r["est"], blocks=r["blocks"], tiles=r["tiles"], kinds=dict(r["kinds"]),
                           **{p: r[p] for p in POL}, shape={str(k): v for k, v in r["shape"].items()}) for n, r in R.items()} |
                  {"T": T, "nslot": m.nslot, "path": a.path}, open(a.json, "w"), indent=1)
