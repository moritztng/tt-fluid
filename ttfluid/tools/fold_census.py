"""What the zero and one-value constant pages make redundant in the outer body, per core and tile pass.

    python -m ttfluid.tools.fold_census EXTRACT.cx.zst [--core 0,7] [--json out.json]

EXTRACT is const_census --slim. For every block (or fused run) of the outer body and every tile pass the pass is
followed symbolically on one core: a constant page that holds one bit pattern is a known value, every other load is
an unknown. Local identities only, no value ranges:

  known op known          -> known (the op folds to a fill)
  x * 1, x / 1, where(known, a, b), x - (+0), x + (-0)      -> an operand, exactly
  x * 0 -> 0, 0 / x -> 0, x + (+0) -> x, x - (-0) -> x, a * 0 + c -> c   -> counted apart (`care`): the bits differ
                                                                            for x < 0, x = -0 or a non-finite x

Then liveness inside the pass from its stores: a store of the value just loaded from the same register is a no-op.
A one-tile L1 register written in a run of several passes is a fused temporary if the pass does not read it before
writing it (that would be an accumulator) and reads it afterwards; otherwise the store is live.
Two results per core: with the exact identities only, and with all of them.
Counted per SIMPLE it: op-tiles, pages by kind and stores that stay live against today's, and passes with no live
store. It is a lower bound of what a per-tile fold removes: `min(acc, 1.0)` or a masked `where` behind an unknown is
not seen.
"""
import argparse
import json
from collections import Counter

import numpy as np

from ttfluid.tools.const_census import load_x

ONE = int(np.float32(1.0).view(np.uint32))
NUM = {"mul": np.multiply, "add": np.add, "sub": np.subtract}


def is0(v):
    return v[0] == "K" and v[1] is not None and v[1] & 0x7fffffff == 0


def is1(v):
    return v[0] == "K" and v[1] == ONE


def one_pass(blks, sig, bt, info, KB, DB, care=True):
    """Counter of one tile pass of a block list; sig: per plain constant read, its page's bit pattern or -1."""
    it = iter(sig)
    nodes, roots, cnt, env = [], [], Counter(), {}
    carry, used = set(), set()      # registers read before written in the pass; temp stores read again

    def node(cat, deps=(), fills=0):
        nodes.append((cat, [d for d in deps if d is not None], fills))
        return len(nodes) - 1

    def rk(r):
        p = info[r]["phys"]
        return "none" if p is None else "const" if p >= KB else "home" if p >= DB else "l1"

    for uops, stores in blks:
        S = {}
        get = lambda s: S.get(s, (("V", ("?", s)), None))
        for u in uops:
            k = u[0]
            if k in ("ld", "lds", "rvf"):
                r = u[2]
                if r in env:
                    v, sn = env[r]
                    used.add(sn)
                    n = node("ld_temp", [sn])
                    S[u[1]] = (v, n if v[0] == "V" else None) if k == "ld" else (("V", ("u", n)), n)
                else:
                    carry.add(r)
                    n = node("ld_l1")
                    S[u[1]] = (("V", ("reg", r) if k == "ld" else ("u", n)), n)
            elif k == "ldr":
                r, plain = u[2], u[3] == 0 and u[4] is None and not u[5]
                if r is None:
                    S[u[1]] = ((("K", 0), None) if plain else (("V", ("u", len(nodes))), node("ldr_zero")))
                    cnt["ldr_zero_plain"] += plain
                elif rk(r) == "const":
                    b = next(it) if plain else -1
                    if b >= 0:
                        cnt["ldr_const_uni"] += 1
                        S[u[1]] = (("K", b), None)
                    else:
                        n = node("ldr_const")
                        S[u[1]] = (("V", ("u", n)), n)
                else:
                    n = node("ldr_home")
                    S[u[1]] = (("V", ("reg", r) if plain else ("u", n)), n)
            elif k == "fill":
                n = node("fill")
                S[u[1]] = (("K", int(np.float32(u[2]).view(np.uint32))), n)
            elif k == "cpd":
                v, s = get(u[1])        # ("cpd", src, dst)
                n = node("cpd", [s], v[0] == "K" and s is None)
                S[u[2]] = (v, n)
            elif k == "un":
                v, s = get(u[2])
                cnt["op_total"] += 1
                if v[0] == "K":
                    cnt["op_const"] += 1
                    S[u[2]] = (("K", None), None)
                else:
                    n = node("op", [s])
                    S[u[2]] = (("V", ("u", n)), n)
            elif k == "fold":
                v, s = get(u[1])
                cnt["op_total"] += 1
                n = node("op", [s], v[0] == "K" and s is None)
                S[u[1]] = (("V", ("u", n)), n)
            elif k == "op":
                name, dst, args = u[1], u[2], [get(a) for a in u[3:]]
                cnt["op_total"] += 1
                vals = [a[0] for a in args]
                res = None      # (value, src, category)
                if name == "where":
                    c = vals[0]
                    if c[0] == "K" and c[1] is not None:
                        res = args[1 if c[1] & 0x7fffffff else 2] + ("op_ident_exact",)
                elif all(v[0] == "K" for v in vals):
                    b = None
                    if name in NUM and all(v[1] is not None for v in vals):
                        f = [np.uint32(v[1]).view(np.float32) for v in vals]
                        with np.errstate(all="ignore"):
                            b = int(np.float32(NUM[name](f[0], f[1])).view(np.uint32))
                    res = (("K", b), None, "op_const")
                elif name == "mul":
                    if is1(vals[0]) or is1(vals[1]):
                        res = args[1 if is1(vals[0]) else 0] + ("op_ident_exact",)
                    elif care and (is0(vals[0]) or is0(vals[1])):
                        res = (("K", 0), None, "op_zero_care")
                elif name == "div":
                    if is1(vals[1]):
                        res = args[0] + ("op_ident_exact",)
                    elif care and is0(vals[0]):
                        res = (("K", 0), None, "op_zero_care")
                elif name == "add":
                    for i in (0, 1):
                        if is0(vals[i]) and (care or vals[i][1]):
                            res = args[1 - i] + ("op_ident_exact" if vals[i][1] else "op_ident_care",)
                            break
                elif name == "sub":
                    if is0(vals[1]) and (care or not vals[1][1]):
                        res = args[0] + ("op_ident_care" if vals[1][1] else "op_ident_exact",)
                elif name in ("mad", "msb"):
                    if care and (is0(vals[0]) or is0(vals[1])):
                        res = args[2] + ("op_ident_care",)
                if res is None:
                    n = node("op", [a[1] for a in args], sum(a[0][0] == "K" and a[1] is None for a in args))
                    S[dst] = (("V", ("u", n)), n)
                else:
                    cnt[res[2]] += 1
                    S[dst] = (res[0], res[1])
            else:
                raise NotImplementedError(k)
        for s, r in stores:
            v, src = get(s)
            kd = rk(r)
            if kd == "l1" and info[r]["size"] == 1 and bt > 1 and r not in carry:
                env[r] = (v, node("st_temp", [src]))
            elif v == ("V", ("reg", r)):
                cnt[f"st_{kd}_noop"] += 1
                cnt[f"st_{kd}_total"] += 1
            else:
                roots.append(node(f"st_{kd}", [src], v[0] == "K" and src is None))
                cnt[f"st_{kd}_total"] += 1
    roots += [n for _, n in env.values() if n not in used]      # a temporary nobody reads in the pass leaves it
    live, todo = set(), list(roots)
    while todo:
        n = todo.pop()
        if n not in live:
            live.add(n)
            todo.extend(nodes[n][1])
    for n, (cat, _, fills) in enumerate(nodes):
        if cat != "op" and (not cat.startswith("st_") or cat == "st_temp"):
            cnt[cat + "_total"] += 1        # op_total and the st_*_total of real stores are counted above
        if n in live:
            cnt[cat + "_live"] += 1
            cnt["fill_new"] += fills
    cnt["pass_total"] += 1
    cnt["pass_dead"] += not roots
    return cnt


def fold(x, core, care=True, log=print):
    info, coff, KB, DB, P = x["info"], x["coff"], x["KBASE"], x["DBASE"], x["pages"][core]
    uni, val = (P == P[:, :1]).all(axis=1), P[:, 0]
    tot, byph = Counter(), {}
    for kind, j, ph, bt, blks, region in x["body"]:
        if region != "outer":
            continue
        plain = [coff[info[u[2]]["const"]] for uops, _ in blks for u in uops
                 if u[0] == "ldr" and u[2] is not None and (info[u[2]]["phys"] or 0) >= KB
                 and u[3] == 0 and u[4] is None and not u[5]]
        cache = {}
        for t in range(bt):
            sig = tuple(int(val[g + t]) if g + t < len(P) and uni[g + t] else -1 for g in plain)
            if sig not in cache:
                cache[sig] = one_pass(blks, sig, bt, info, KB, DB, care)
            tot.update(cache[sig])
            byph.setdefault(ph, Counter()).update(cache[sig])

    def row(c):
        # op_tiles: un / fold / op uops of today's pass; op_live: those a live store still needs; cpd: tile copies
        return dict(passes=c["pass_total"], passes_dead=c["pass_dead"], op_tiles=c["op_total"],
                    op_fold_const=c["op_const"], op_ident_exact=c["op_ident_exact"], op_ident_care=c["op_ident_care"],
                    op_zero_care=c["op_zero_care"], op_live=c["op_live"], cpd=c["cpd_total"], cpd_live=c["cpd_live"],
                    const_pages=c["ldr_const_total"] + c["ldr_const_uni"], const_pages_uniform=c["ldr_const_uni"],
                    const_pages_live=c["ldr_const_live"], home_pages=c["ldr_home_total"], home_pages_live=c["ldr_home_live"],
                    l1_loads=c["ld_l1_total"] + c["ld_temp_total"], l1_loads_live=c["ld_l1_live"] + c["ld_temp_live"],
                    st_home=c["st_home_total"], st_home_noop=c["st_home_noop"], st_l1=c["st_l1_total"], st_l1_noop=c["st_l1_noop"],
                    st_temp=c["st_temp_total"], st_temp_live=c["st_temp_live"], fills_today=c["fill_total"],
                    fills_live=c["fill_live"], fills_new=c["fill_new"])
    out = dict(core=core, care=care, all=row(tot), by_phase={p: row(c) for p, c in byph.items()})
    a = out["all"]
    live_ops = tot["op_live"]
    log(f"core {core}, {'all identities' if care else 'exact identities only'}: {a['passes']} tile passes per it, "
        f"{a['passes_dead']} with no live store")
    log(f"  op-tiles {a['op_tiles']}: fold to a fill {a['op_fold_const']}, exact identities {a['op_ident_exact']}, "
        f"identities up to the sign of zero {a['op_ident_care']}, x*0 {a['op_zero_care']}; live after the fold {live_ops}, "
        f"copies {a['cpd']} -> {a['cpd_live']}, new fills {a['fills_new']}")
    log(f"  constant pages {a['const_pages']}: one-value {a['const_pages_uniform']}, live after the fold {a['const_pages_live']}")
    log(f"  DRAM vector pages read {a['home_pages']}: live {a['home_pages_live']}; written {a['st_home']}: no-op {a['st_home_noop']}")
    log(f"  L1 loads {a['l1_loads']}: live {a['l1_loads_live']}; L1 stores {a['st_l1']} (no-op {a['st_l1_noop']}), "
        f"fused temporaries {a['st_temp']} (live {a['st_temp_live']})")
    for p, r in out["by_phase"].items():
        if r["op_tiles"] > 1000:
            log(f"  {p}: op-tiles {r['op_tiles']} -> live {byph[p]['op_live']}; const pages {r['const_pages']} -> {r['const_pages_live']}; "
                f"DRAM vector pages read {r['home_pages']} -> {r['home_pages_live']}; passes {r['passes']}, dead {r['passes_dead']}")
    out["op_live"] = int(live_ops)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extract")
    ap.add_argument("--core", default="0")
    ap.add_argument("--json")
    a = ap.parse_args()
    x = load_x(a.extract)
    res = [fold(x, int(c), care) for c in a.core.split(",") if int(c) in x["pages"] for care in (False, True)]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(src=a.extract, cores=res), f)


if __name__ == "__main__":
    main()
