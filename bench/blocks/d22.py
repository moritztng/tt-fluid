"""flx-kernel d22.py: D22's number, "assembly ops, cycles per tile-it", from a run_mesh --prof leg.

  static (CPU, on the box, PYTHONPATH = the tree):
      python3 d22.py --static MCACHE.pkl[.zst] --out S.json
          per region of the machine program (top, outer = the assembly, solve0..) and per census slot: op-tiles per
          core for ONE execution of the region, for arm A (stock op words) and arm B (generated bodies + the stock
          words left), the priced cycles (isolated unit costs, blockgen.STOCK; a body = 17 + 32 L + its pushed words)
  leg (any python3, no imports beyond the stdlib):
      python3 d22.py LEG.jsonl [LEG.log] [LEG2.jsonl LEG2.log] [--its N] [--T T] [--S S.json] [--scale F]

What the profile build measures (vm_compute.cpp CFD_PROF): every interpreter thread adds the wall-clock cycles from
one op's issue to the next op's issue to the slot of that op (40 slots, whole run, all phases; stored in 16-cycle
units), and UNPACK counts the ops. C_BLK (46) is counted in slot 0 with the run's one C_END; ops 40-45 (STW, MAD,
MSB, LDRH, STWH, PH) share slot 39 with LDR. run_mesh reports the mean over cores (jsonl "census").
A slot holds whatever the thread waited for before the next op could issue. The slots are not split by phase, so
the assembly's part (region "outer": the outer loop without the Krylov loops) is taken by counts: the machine gives
the op-tiles of one pass of every region, the outer body runs its times, the Krylov loops' passes are fitted to
the counted ops (and printed next to the its of the run), and a slot's cycles are split in proportion (stock slots
by op-tiles, the body slot by the bodies' priced cycles: bodies differ in length).
Per tile-it = cycles of one core / (T x its); per cell-it = / 1024.
"""
import json, re, sys

CN = ["END", "ACQ", "COMMIT", "REL", "LD", "ST", "FILL", "RVF", "add", "sub", "mul", "div", "max", "min", "WHERE",
      "CPD", "neg", "abs", "sqrt", "tanh", "log", "exp", "gtz", "gez", "ltz", "lez", "eqz", "nez", "FOLD", "TOK",
      "SIGB", "SIGN", "WAITB", "WAITN", "JMP", "BRT", "LDS", "TBEG", "TEND", "LDR+STW"]
ARITH = ["FILL", "add", "sub", "mul", "div", "max", "min", "WHERE", "CPD", "neg", "abs", "sqrt", "tanh", "log", "exp",
         "gtz", "gez", "ltz", "lez", "eqz", "nez", "FOLD"]          # what blockgen.stock_cost prices (fill, cpd, fold included)
LOADS = ["LD", "LDS", "RVF", "LDR+STW", "ST"]
BUDGET_TILE, BUDGET_CELL = 1.0e6, 8.8e3     # D22's rule per tile-it; D28's whole budget per cell-it per core


def arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


# ---- static: the machine's op-tiles per region ---------------------------------------------------------------------
def static(path, out, rec=1.0):
    import collections, pickle
    from ttfluid.resident import blklower, blockgen
    from ttfluid.resident.census import region_of

    def load(p):
        if p.endswith(".zst"):
            import zstandard
            with open(p, "rb") as f:
                return pickle.load(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True))
        with open(p, "rb") as f:
            return pickle.load(f)

    def slot(u):
        k = u[0]
        if k == "body":
            return "BODY"
        if k in ("op", "un"):
            return "WHERE" if u[1] == "where" else "LDR+STW" if u[1] in ("mad", "msb") else u[1]
        return {"ld": "LD", "lds": "LDS", "ldr": "LDR+STW", "fill": "FILL", "rvf": "RVF", "cpd": "CPD", "fold": "FOLD"}.get(k, k)

    saved, prog, m = load(path)
    T, D = m.T, getattr(m, "dram", ())
    reg = region_of(m.mir)
    R = collections.defaultdict(lambda: dict(A=collections.Counter(), B=collections.Counter(), A_priced=0.0, B_left_priced=0.0,
                                             B_body_model=0.0, B_body_instr=0.0, B_body_words=0.0, blocks=0, block_tiles=0.0))
    for j, ins in enumerate(m.mir):
        if ins[0] not in ("blk", "seg"):
            continue
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]
        bt = m.tiles(blks)
        r = R[reg[j]]
        for b in blks:
            vec = any(u[0] == "ldr" or (u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__" and m.is_vec(u[2])) for u in b[1]) \
                or any(m.is_vec(x) for _, x in b[2])
            w = bt if (ins[0] == "seg" or vec) else 1      # op-tiles of this block per core per pass of its region
            r["blocks"] += 1
            r["block_tiles"] += w
            for u in b[1]:
                r["A"][slot(u)] += w
            for _, x in b[2]:
                r["A"]["LDR+STW" if x in D else "ST"] += w
            r["A_priced"] += w * blockgen.stock_cost(b[1])
            items, st2 = blklower.lower_block(b[1], b[2], m.nslot, accept=blockgen.accept)
            for u in items:
                r["B"][slot(u)] += w
                if u[0] == "body":
                    cfg, row = blockgen.program(u[1], sym=True)
                    L = len(row)
                    words = 1 + len(cfg) + L * (1 if L >= 4 else 2)
                    r["B_body_instr"] += w * L
                    r["B_body_words"] += w * words
                    r["B_body_model"] += w * (17 + 32 * L + rec * words)
            for _, x in st2:
                r["B"]["LDR+STW" if x in D else "ST"] += w
            r["B_left_priced"] += w * blockgen.stock_cost([u for u in items if u[0] != "body"])
    S = dict(mcache=path, T=T, nslot=m.nslot, rec=rec,
             regions={k: dict(v, A=dict(v["A"]), B=dict(v["B"])) for k, v in R.items()})
    json.dump(S, open(out, "w"), indent=1)
    M = 1e-6
    print("static %s: T %d, regions %s -> %s" % (path, T, sorted(R), out))
    for k in sorted(R):
        v = R[k]
        print("  %-7s blocks %5d, block tiles per pass %9.1f | priced per pass, M cycles per tile: A stock words %.3f | "
              "B bodies %.3f (%.0f bodies, %.1f instr a body) + stock left %.3f = %.3f"
              % (k, v["blocks"], v["block_tiles"], v["A_priced"] * M / T, v["B_body_model"] * M / T, v["B"].get("BODY", 0),
                 v["B_body_instr"] / max(v["B"].get("BODY", 0), 1), v["B_left_priced"] * M / T,
                 (v["B_body_model"] + v["B_left_priced"]) * M / T))


# ---- leg: the measured slots -----------------------------------------------------------------------------------------
def solve(A, y):
    """Least squares x of A x = y (rows = equations), normal equations, Gauss elimination; None when singular."""
    n = len(A[0]) if A else 0
    N = [[sum(A[r][i] * A[r][j] for r in range(len(A))) for j in range(n)] + [sum(A[r][i] * y[r] for r in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(N[r][i]))
        if abs(N[p][i]) < 1e-9:
            return None
        N[i], N[p] = N[p], N[i]
        for r in range(n):
            if r != i:
                f = N[r][i] / N[i][i]
                N[r] = [a - f * b for a, b in zip(N[r], N[i])]
    return [N[i][n] / N[i][i] for i in range(n)]


def leg(path, log, its, T, S, scale):
    r = [json.loads(x) for x in open(path) if x.strip()][-1]
    c = r["census"]
    T = T or r["T"]
    its = its or int(r.get("n") or r["last"]["outer"])
    blocks = int(r.get("blocks") or 0)
    if not blocks and log:
        m_ = re.search(r"--blocks (\d)", open(log, errors="replace").read(4000))
        blocks = int(m_.group(1)) if m_ else 0
    arm = "B" if blocks else "A"
    ops = {k: dict(v) for k, v in c["math"]["ops"].items()}
    if blocks and "END" in ops:          # slot 0: the run's one C_END and every C_BLK body
        e = ops.pop("END")
        ops["BODY"] = dict(cyc=e["cyc"], n=e["n"] - 1, per=e["cyc"] / max(e["n"] - 1, 1))
    den = T * its
    hz = max(x["max"] for x in r["aiclk"]) * 1e6 if r.get("aiclk") else 1350e6
    tot = c["math"]["total"]
    print("%s: arm %s (--blocks %d), T %d, its %d, K %d, ms_device %s, AICLK in-process %s, MATH %.4g cycles a core (%.2f s)"
          % (path, arm, blocks, T, its, r.get("K", 1), ",".join("%.1f" % x for x in r.get("ms_device", [])),
             r.get("aiclk"), tot, tot / hz))
    names = ([x for x in ["BODY"] if x in ops] + [x for x in ARITH if x in ops])
    grp = lambda L: sum(ops[k]["cyc"] for k in L if k in ops)
    ctrl = [k for k in ops if k not in names and k not in LOADS]
    print("  whole run (assembly + solves + init), MATH thread, per tile-it: op words %.3f M (%.1f %% of MATH), loads and stores %.3f M, "
          "control (ACQ COMMIT REL TBEG TEND ...) %.3f M, all %.3f M"
          % (grp(names) / den / 1e6, 100 * grp(names) / tot, grp(LOADS) / den / 1e6, grp(ctrl) / den / 1e6, tot / den / 1e6))
    passes, fit = None, None
    if S:
        Rg = S["regions"]
        sol = sorted(k for k in Rg if k.startswith("solve"))
        cnt = lambda reg, s: Rg.get(reg, {}).get(arm, {}).get(s, 0.0)
        F = [s for s in names + ["LD", "LDS", "ST", "LDR+STW", "RVF"] if s in ops and any(cnt(k, s) for k in sol)]
        y = [ops[s]["n"] - its * cnt("outer", s) - cnt("top", s) for s in F]
        A = [[cnt(k, s) for k in sol] for s in F]
        x = solve(A, y) if F and sol else None
        if x is not None:
            passes = dict(zip(sol, x))
            pred = {s: its * cnt("outer", s) + cnt("top", s) + sum(passes[k] * cnt(k, s) for k in sol) for s in ops}
            fit = max((abs(pred[s] - ops[s]["n"]) / max(ops[s]["n"], 1), s) for s in F)
            rows = r.get("rows") if isinstance(r.get("rows"), list) else []
            seen = {k: sum(row.get("its%d" % i, 0) for row in rows) for i, k in enumerate(sol)}
            print("  Krylov passes fitted to the op counts: %s | its summed over the run's %d rows: %s | worst count residual %.2f %% (%s)"
                  % ({k: round(v, 1) for k, v in passes.items()}, len(rows), {k: round(v) for k, v in seen.items()} if rows else "-",
                     100 * fit[0], fit[1]))
        else:      # e.g. five solve loops with the same op mix: their passes cannot be told apart by counts
            rows = r.get("rows") if isinstance(r.get("rows"), list) else []
            if rows and sol:
                passes = {k: float(sum(row.get("its%d" % i, 0) for row in rows)) for i, k in enumerate(sol)}
                pred = {s: its * cnt("outer", s) + cnt("top", s) + sum(passes[k] * cnt(k, s) for k in sol) for s in ops}
                fit = max((abs(pred[s] - ops[s]["n"]) / max(ops[s]["n"], 1), s) for s in F) if F else (0.0, "-")
                print("  Krylov passes = the its of the run's %d rows (the count fit is singular: equal solve loops): %s | worst count residual %.2f %% (%s)"
                      % (len(rows), {k: round(v) for k, v in passes.items()}, 100 * fit[0], fit[1]))
            else:
                print("  no Krylov passes (no rows): stock slots are split by the counted ops, the BODY slot is NOT split (upper bound)")
    print("  %-8s %12s %10s %14s %14s %9s" % ("slot", "op-tiles", "cyc each", "whole run", "assembly", "share"))
    print("  %-8s %12s %10s %14s %14s %9s" % ("", "per tile-it", "in situ", "cyc/tile-it", "cyc/tile-it", "assembly"))
    asm_tot = 0.0
    out = {}
    for s in names:
        o = ops[s]
        share = None
        if S:
            Rg = S["regions"]
            if s == "BODY":
                w = lambda reg: Rg.get(reg, {}).get("B_body_model", 0.0)
            else:
                w = lambda reg, s=s: Rg.get(reg, {}).get(arm, {}).get(s, 0.0)
            a = its * w("outer")
            b = w("top") + sum((passes or {}).get(k, 0.0) * w(k) for k in Rg if k.startswith("solve"))
            if s != "BODY":                          # stock slots: the outer body's op-tiles over the counted ones
                share = min(1.0, its * w("outer") / max(o["n"], 1))
            else:
                share = a / (a + b) if a + b else 0.0
        asm = o["cyc"] * share if share is not None else None
        if asm is not None:
            asm_tot += asm
        out[s] = asm
        print("  %-8s %12.1f %10.1f %14.0f %14s %9s" % (s, o["n"] / den, o["cyc"] / max(o["n"], 1), o["cyc"] / den,
                                                       "%.0f" % (asm / den) if asm is not None else "-",
                                                       "%.1f %%" % (100 * share) if share is not None else "-"))
    whole = grp(names) / den
    print("  op words, whole run: %.3f M cycles per tile-it = %.2fk per cell-it" % (whole / 1e6, whole / 1024 / 1e3))
    if S:
        a = asm_tot / den
        Rg = S["regions"]["outer"]
        priced = (Rg["A_priced"] if arm == "A" else Rg["B_body_model"] + Rg["B_left_priced"]) / T
        print("  ASSEMBLY OPS (region outer), measured in situ: %.3f M cycles per tile-it (D22's rule %.1f M: %s) = %.2fk per cell-it of the %.1fk budget"
              % (a / 1e6, BUDGET_TILE / 1e6, "UNDER" if a < BUDGET_TILE else "OVER", a / 1024 / 1e3, BUDGET_CELL / 1e3))
        print("  the same machine priced with isolated unit costs: %.3f M per tile-it; in situ / isolated %.2f"
              % (priced / 1e6, a / priced if priced else 0))
        if scale:
            print("  on the hood proxy's scale (x %.3f = priced proxy / priced this machine): %.3f M cycles per tile-it"
                  % (scale, a * scale / 1e6))
    ph = r.get("phase") or {}
    if ph:
        A_ = [k for k in ph if k in ("mom", "p_asm", "flux", "turb_asm")]
        s_asm = sum(ph[k]["med"] for k in A_)
        s_all = sum(v["med"] for v in ph.values())
        print("  phases (MATH wall, median of cores): assembly %s %.4f s of %.4f s = %.3f M cycles per tile-it for everything the "
              "MATH thread does and waits for there" % ("+".join(A_), s_asm, s_all, s_asm * hz / den / 1e6))
    return dict(arm=arm, T=T, its=its, whole=whole, asm=asm_tot / den if S else None)


if __name__ == "__main__":
    if "--static" in sys.argv:
        p = arg("--static")
        static(p, arg("--out", "static.json"), arg("--rec", 1.0, float))
        sys.exit(0)
    its, T, sp, scale = arg("--its", None, int), arg("--T", None, int), arg("--S"), arg("--scale", None, float)
    S = json.load(open(sp)) if sp else None
    files = sys.argv[1:]
    js = [f for f in files if f.endswith(".jsonl")]
    lg = [f for f in files if not f.endswith(".jsonl")]
    res = []
    for i, f in enumerate(js):
        res.append(leg(f, lg[i] if i < len(lg) else None, its, T, S, scale))
        print()
    if len(res) == 2 and res[0]["arm"] != res[1]["arm"]:
        a, b = (res[0], res[1]) if res[0]["arm"] == "A" else (res[1], res[0])
        print("A -> B: op words whole run %.3f -> %.3f M per tile-it (%.2fx)" % (a["whole"] / 1e6, b["whole"] / 1e6, b["whole"] / a["whole"])
              + ("; assembly ops %.3f -> %.3f M per tile-it (%.2fx)" % (a["asm"] / 1e6, b["asm"] / 1e6, b["asm"] / a["asm"]) if a["asm"] and b["asm"] else ""))
