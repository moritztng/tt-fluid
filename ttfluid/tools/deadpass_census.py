"""Dead tile-pass propagation census (lever 27): how much of the program is dead once zero/uniform constant pages are known.

    python -m ttfluid.tools.deadpass_census EXTRACT.cx.zst [--aux AUX.npz] [--core 0,7] [--json out.json]
    python -m ttfluid.tools.deadpass_census IMG_DIR [--core 0,55,110,165] [--json out.json]
    python -m ttfluid.tools.deadpass_census --self-test

Accepts either a const_census extract (*.cx.zst, with optional sidecar aux.npz for per-tile gather counts and
pre-initialized read-only tiles) or a saved device Image directory (shared.npz + chipK/*.npy, memory-mapped). Both
frontends decode into the same canonical IR (outer body and every inner solver loop solve0..solveK) and run the same
inter-instruction, per-tile propagation and backward liveness pass.

Propagation rules (both `exact` [bitwise-safe] and `all` [algebraic identities, default]):
  known op known              -> known (exact fp32 bits)
  x * 1, x / 1, where(K,a,b), where(c,a,a), x - (+0), x + (-0) -> operand (exact)
  x * 0 -> +0, 0 / x -> +0, y + (+0) -> y, y - (-0) -> y, mad/msb(x, 0, c) -> c (algebraic; breaks -0 / NaN*0)
  a gather or stream read whose only consumers are dead -> dead
  a store of a zero/uniform or copy value -> dead if every live reader can take the constant or original un-clobbered
  source without adding DRAM traffic and the slot is not live-out; otherwise kept live with its blocking reason logged.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ttfluid.resident import encode as E
from ttfluid.resident.machine import DBASE, LANES
from ttfluid.resident.program import PHASES

ONE_U32 = int(np.float32(1.0).view(np.uint32))
ZERO_POS = 0x00000000
ZERO_NEG = 0x80000000
DENSE_BIT = 1 << 31

C_BIN_REV = {v: k for k, v in E.C_BIN.items()}
C_UN_REV = {v: k for k, v in E.C_UN.items()}

UN_FN = {
    "neg": np.negative,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "log": np.log,
    "exp": np.exp,
    "gtz": lambda x: np.float32(1.0 if x > 0 else 0.0),
    "gez": lambda x: np.float32(1.0 if x >= 0 else 0.0),
    "ltz": lambda x: np.float32(1.0 if x < 0 else 0.0),
    "lez": lambda x: np.float32(1.0 if x <= 0 else 0.0),
    "eqz": lambda x: np.float32(1.0 if x == 0 else 0.0),
    "nez": lambda x: np.float32(1.0 if x != 0 else 0.0),
}
BIN_FN = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
}


def u32_to_f32(b):
    return np.uint32(b).view(np.float32)


def f32_to_u32(x):
    return int(np.float32(x).view(np.uint32))


def is_nan_u32(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def is_zero_u32(b):
    return (b & 0x7FFFFFFF) == 0


def eval_un_u32(name, b):
    with np.errstate(all="ignore"):
        y = UN_FN[name](u32_to_f32(b))
    res = f32_to_u32(y)
    return None if is_nan_u32(res) else res


def eval_bin_u32(name, a, b):
    with np.errstate(all="ignore"):
        y = BIN_FN[name](u32_to_f32(a), u32_to_f32(b))
    res = f32_to_u32(y)
    return None if is_nan_u32(res) else res


def eval_mad_u32(name, a, b, c):
    with np.errstate(all="ignore"):
        pr = np.float64(u32_to_f32(a)) * np.float64(u32_to_f32(b))
        cv = np.float64(u32_to_f32(c))
        y = np.float32(cv + pr if name == "mad" else cv - pr)
    res = f32_to_u32(y)
    return None if is_nan_u32(res) else res


def eval_fold_u32(b):
    if is_zero_u32(b):
        return b
    # Exact 32x32 column-then-row fp32 sum (matches memu.py line 428)
    with np.errstate(all="ignore"):
        v = u32_to_f32(b)
        row = np.float32(0.0)
        for _ in range(32):
            row = np.float32(row + v)
        tot = np.float32(0.0)
        for _ in range(32):
            tot = np.float32(tot + row)
    res = f32_to_u32(tot)
    return None if is_nan_u32(res) else res


def page_uniform_u32(pages_u32):
    """For (N, 1024) uint32 pages, return int64 array: bit pattern if all 1024 lanes match and are non-NaN, else -1."""
    if len(pages_u32) == 0:
        return np.zeros(0, np.int64)
    first = pages_u32[:, 0]
    uni = (pages_u32 == first[:, None]).all(axis=1)
    is_nan = ((first & 0x7F800000) == 0x7F800000) & ((first & 0x007FFFFF) != 0)
    out = np.full(len(pages_u32), -1, np.int64)
    keep = uni & ~is_nan
    out[keep] = first[keep].astype(np.int64)
    return out


# ==================================================================================================
# Frontend 1: const_census extract (*.cx.zst) + optional sidecar aux.npz
# ==================================================================================================

def load_from_extract(cx_path, core, aux_path=None):
    from ttfluid.tools.const_census import load_x

    x = load_x(cx_path)
    assert core in x["pages"], f"core {core} not in extract (have {sorted(x['pages'])})"
    T = int(x["T"])
    KB, DB = int(x["KBASE"]), int(x["DBASE"])
    info, coff = x["info"], x["coff"]
    const_uni = page_uniform_u32(x["pages"][core])

    def reg_desc(r):
        if r is None:
            return None
        p = info[r]["phys"]
        if p >= KB:
            return ("CONST", coff[info[r]["const"]])
        if p >= DB:
            return ("HOME", p - DB)
        return ("L1", p)

    def reg_loc(r):
        p = info[r]["phys"]
        vec = bool(info[r]["vec"])
        if p >= DB:
            return ("HOME", p - DB, True)
        return ("L1", p, vec)

    # Merge body and rest by MIR index j
    events = []
    for kind, j, ph, bt, blks, reg in x["body"]:
        cblks = []
        for uops, stores in blks:
            cu = []
            for u in uops:
                k = u[0]
                if k == "ld":
                    cu.append(("ld", u[1], reg_loc(u[2])))
                elif k == "lds":
                    cu.append(("lds", u[1], reg_loc(u[2]), int(u[3])))
                elif k == "rvf":
                    cu.append(("rvf", u[1], reg_loc(u[2])))
                elif k == "ldr":
                    r = u[2]
                    src_desc = reg_desc(r)
                    parts = [(reg_desc(y), h) for y, h in u[5]]
                    cu.append(("ldr", u[1], src_desc, int(u[3]), u[4], parts, False, False))
                elif k == "fill":
                    cu.append(("fill", u[1], f32_to_u32(u[2])))
                elif k == "cpd":
                    cu.append(("cpd", u[1], u[2]))
                elif k == "op":
                    cu.append(("op", u[1], u[2], *u[3:]))
                elif k == "un":
                    cu.append(("un", u[1], u[2]))
                elif k == "fold":
                    cu.append(("fold", u[1]))
                else:
                    raise NotImplementedError(k)
            cs = [(s, reg_loc(r)) for s, r in stores if r != "__anchor__"]
            cblks.append((cu, cs))
        # Scalar block check (matches memu.py line 169)
        vec_any = False
        for cu, cs in cblks:
            for u in cu:
                if u[0] == "ldr":
                    vec_any = True
                elif u[0] in ("ld", "lds", "rvf") and u[2][2]:
                    vec_any = True
            for _, loc in cs:
                if loc[2]:
                    vec_any = True
        eff_bt = bt if vec_any else 1
        events.append((j, reg, ph, ("blk_run", eff_bt, cblks)))

    state_locs = set()
    for j, ph, reg, ins in x["rest"]:
        op = ins[0]
        if op == "gather":
            dst_r, mid, parts = ins[1], ins[2], ins[3]
            mode = ins[4] if len(ins) > 4 else 0
            # Note: gather dst may not be in info if never read in a blk (rare), check info
            if dst_r in info:
                dst_l1 = info[dst_r]["phys"]
                assert dst_l1 < DB
                cparts = [(reg_desc(y), h) for y, h in parts]
                events.append((j, reg, ph, ("gather", dst_l1, int(mid), cparts, int(mode))))
        elif op == "fetch":
            dst_r, c_r = ins[1], ins[2]
            if dst_r in info:
                events.append((j, reg, ph, ("fetch", info[dst_r]["phys"], coff[c_r], info[dst_r]["size"])))
        elif op == "halo":
            h, src_r = ins[1], ins[2]
            if src_r in info:
                events.append((j, reg, ph, ("halo", int(h), reg_desc(src_r))))
        elif op in ("red", "red2"):
            pairs = []
            for fr, pr in (zip(ins[1::2], ins[2::2]) if op == "red2" else [(ins[1], ins[2])]):
                if pr in info and fr in info:
                    pairs.append((info[pr]["phys"], info[fr]["phys"]))
            events.append((j, reg, ph, ("red", pairs)))
        elif op == "out":
            if ins[2] in info:
                events.append((j, reg, ph, ("out", ins[1], info[ins[2]]["phys"])))
        elif op == "jumpif":
            if ins[1] in info:
                events.append((j, reg, ph, ("branch", info[ins[1]]["phys"])))
        elif op == "keep":
            for r in ins[1]:
                if r in info:
                    p, sz = info[r]["phys"], info[r]["size"]
                    sp = "HOME" if p >= DB else "L1"
                    base = p - DB if p >= DB else p
                    for t in range(sz):
                        state_locs.add((sp, base + t))

    events.sort(key=lambda e: e[0])

    # Load optional aux.npz (map entry counts and pre-initialized read-only tiles)
    map_counts = {}
    init_ro = {}
    if aux_path:
        az = np.load(aux_path)
        acores = az["cores"].tolist()
        assert core in acores, f"core {core} not in aux {acores}"
        ci = acores.index(core)
        lc, hc, zc = az["loc_cnt"][ci], az["halo_cnt"][ci], az["zero_cnt"][ci]
        for mid in range(lc.shape[0]):
            map_counts[mid] = (lc[mid].astype(np.int64), hc[mid].astype(np.int64), zc[mid].astype(np.int64), False)
        pre = json.loads(bytes(az["pre_info"]).decode())
        for _, v in pre.items():
            p = v["phys"]
            sp = "HOME" if p >= DB else "L1"
            base = p - DB if p >= DB else p
            for ti, bits in enumerate(v["uni"][str(core)]):
                if bits >= 0 and not is_nan_u32(bits):
                    init_ro[(sp, base + ti)] = int(bits)
        st = json.loads(bytes(az["state_regs"]).decode())
        for s in st:
            p, sz = s["phys"], s["size"]
            sp = "HOME" if p >= DB else "L1"
            base = p - DB if p >= DB else p
            for t in range(sz):
                state_locs.add((sp, base + t))

    return dict(
        name=Path(cx_path).name,
        core=core,
        T=T,
        cells=int(x["cells"][core]),
        cpages=int(x["cpages"]),
        const_uni=const_uni,
        init_ro=init_ro,
        state_locs=state_locs,
        map_counts=map_counts,
        events=events,
        flags=x.get("lower", {}),
    )


# ==================================================================================================
# Frontend 2: Encoded device Image directory (shared.npz + chipK/*.npy)
# ==================================================================================================

def _decode_desc(d):
    kind = (d >> 28) & 0xF
    narrow = bool(d & E.NARROW)
    kfold = bool(d & E.FOLD)
    val = d & ((1 << 26) - 1)
    if kind == 3:
        return None, narrow, kfold
    if kind == 2:
        return ("CONST", val), narrow, kfold
    if kind == 1:
        return ("HOME", val), narrow, kfold
    if kind == 0:
        return ("L1", val), narrow, kfold
    raise ValueError(f"bad desc {d:#x}")


def _parse_b_stream(b, T):
    ops = []
    pc, n = 0, len(b)
    while pc < n:
        w = int(b[pc])
        op, f0 = w & 0xFF, (w >> 8) & 0xF
        if op == E.B_END:
            ops.append((pc, "END"))
            pc += 1
            break
        elif op == E.B_SIGC:
            ops.append((pc, "SIGC")); pc += 1
        elif op == E.B_WAITC:
            ops.append((pc, "WAITC")); pc += 1
        elif op == E.B_SIGN:
            ops.append((pc, "SIGN")); pc += 1
        elif op == E.B_WAITN:
            ops.append((pc, "WAITN")); pc += 1
        elif op == E.B_HALO:
            desc, _, _ = _decode_desc(int(b[pc + 2]))
            ops.append((pc, "HALO", int(b[pc + 1]), desc))
            pc += 3
        elif op == E.B_RED:
            cnt = f0 + 1
            pairs = [(int(b[pc + 1 + 2 * k]), int(b[pc + 2 + 2 * k])) for k in range(cnt)]
            ops.append((pc, "RED", pairs))
            pc += 1 + 2 * cnt
        elif op == E.B_OUT:
            ops.append((pc, "OUT", int(b[pc + 1]), int(b[pc + 2])))
            pc += 3
        elif op == E.B_SINK:
            cnt = int(b[pc + 1]) & 0xFFFF
            tn = (int(b[pc + 1]) >> 16) or T
            sinks = []
            for k in range(cnt):
                desc, narrow, _ = _decode_desc(int(b[pc + 2 + k]))
                sinks.append((desc, narrow))
            ops.append((pc, "SINK", tn, sinks))
            pc += 2 + cnt
        elif op == E.B_FETCH:
            ops.append((pc, "FETCH", int(b[pc + 1]), int(b[pc + 2]), int(b[pc + 3])))
            pc += 4
        elif op == E.B_SCAL:
            nw = int(b[pc + 1])
            ops.append((pc, "SCAL", b[pc + 2:pc + 2 + nw].tolist()))
            pc += 2 + nw
        elif op == E.B_GATHER:
            dst, mid = int(b[pc + 1]), int(b[pc + 2])
            parts = [(("L1", int(b[pc + 3 + k])), None) for k in range(f0)]
            ops.append((pc, "BGATHER", dst, mid, parts))
            pc += 3 + f0
        elif op == E.B_JMP:
            ops.append((pc, "JMP", int(b[pc + 1])))
            pc += 2
        elif op == E.B_BRT:
            ops.append((pc, "BRT", int(b[pc + 1]), int(b[pc + 2])))
            pc += 3
        elif op == E.B_PH:
            ops.append((pc, "PH", f0))
            pc += 1
        else:
            raise ValueError(f"bad B op {op} at pc={pc}")
    return ops


def _parse_n_stream(n_arr, T, wide=False):
    ops = []
    pc, length = 0, len(n_arr)
    ss = 14 if wide else 16
    sm = 0x3FF if wide else 0xFFF
    sh_bits = 32 - ss
    sh_sign = 1 << (sh_bits - 1)
    sh_mask = (1 << sh_bits) - 1
    while pc < length:
        w = int(n_arr[pc])
        op, f0 = w & 0xFF, (w >> 8) & 0xF
        if op == E.N_END:
            ops.append((pc, "END"))
            pc += 1
            break
        elif op == E.N_SIGC:
            ops.append((pc, "SIGC")); pc += 1
        elif op == E.N_WAITC:
            ops.append((pc, "WAITC")); pc += 1
        elif op == E.N_SIGB:
            ops.append((pc, "SIGB")); pc += 1
        elif op == E.N_WAITB:
            ops.append((pc, "WAITB")); pc += 1
        elif op == E.N_GATHER:
            mode = (w >> 12) & 0xF
            gt = (w >> 16) & 0xFF
            dst, mid = int(n_arr[pc + 1]), int(n_arr[pc + 2])
            parts = []
            for k in range(f0):
                pdesc, _, _ = _decode_desc(int(n_arr[pc + 3 + 2 * k]))
                hs = int(n_arr[pc + 4 + 2 * k])
                parts.append((pdesc, None if hs == E.NONE16 else hs))
            ops.append((pc, "GATHER", dst, mid, parts, mode, gt))
            pc += 3 + 2 * f0
        elif op == E.N_HALO:
            desc, _, _ = _decode_desc(int(n_arr[pc + 2]))
            ops.append((pc, "HALO", int(n_arr[pc + 1]), desc))
            pc += 3
        elif op == E.N_STREAM:
            cnt = int(n_arr[pc + 1]) & 0xFFFF
            tn = (int(n_arr[pc + 1]) >> 16) or T
            ents = []
            p = pc + 2
            for _ in range(cnt):
                src_desc, narrow, kfold = _decode_desc(int(n_arr[p]))
                x = int(n_arr[p + 1])
                sh_raw = (x >> ss) & sh_mask
                shift = sh_raw - (1 << sh_bits) if (sh_raw & sh_sign) else sh_raw
                mid1 = (x >> 4) & sm
                mid = None if mid1 == 0 else mid1 - 1
                npart = x & 0xF
                parts = []
                for k in range(npart):
                    pdesc, _, _ = _decode_desc(int(n_arr[p + 2 + 2 * k]))
                    hs = int(n_arr[p + 3 + 2 * k])
                    parts.append((pdesc, None if hs == E.NONE16 else hs))
                ents.append((src_desc, shift, mid, parts, narrow, kfold))
                p += 2 + 2 * npart
            ops.append((pc, "STREAM", tn, ents))
            pc = p
        elif op == E.N_CODE:
            pc += 3
        elif op == E.N_CODEAT:
            pc += 2
        elif op == E.N_JMP:
            ops.append((pc, "JMP", int(n_arr[pc + 1])))
            pc += 2
        elif op == E.N_BRT:
            ops.append((pc, "BRT", int(n_arr[pc + 1]), int(n_arr[pc + 2])))
            pc += 3
        else:
            raise ValueError(f"bad N op {op} at pc={pc}")
    return ops


def _parse_c_block(c, pc, paged=True):
    """Parse one C_ACQ ... C_COMMIT ... C_REL block starting at pc (pointing at C_ACQ)."""
    assert (int(c[pc]) & 0x7F) == E.C_ACQ
    pc += 1
    uops = []
    stores = []
    in_store = False
    while True:
        w = int(c[pc])
        op = w & 0x7F
        is_tile = bool(w & E.TILE)
        f0, f1, f2, f3 = (w >> 8) & 0xF, (w >> 12) & 0xF, (w >> 16) & 0xF, (w >> 20) & 0xF
        if paged and op == E.C_JMP:
            pc = (pc // E.PW + 1) * E.PW
            continue
        if op == E.C_COMMIT:
            in_store = True
            pc += 1
        elif op == E.C_REL:
            pc += 1
            return uops, stores, pc
        elif not in_store:
            if op == E.C_LD:
                uops.append(("ld", f0, ("L1", w >> 16, is_tile)))
                pc += 1
            elif op == E.C_LDS:
                sh_q = int(np.int32(c[pc + 1]))
                uops.append(("lds", f0, ("L1", w >> 16, is_tile), sh_q * 4))
                pc += 2
            elif op == E.C_RVF:
                uops.append(("rvf", f0, ("L1", w >> 16, False)))
                pc += 1
            elif op in (E.C_LDR, E.C_LDRH, E.C_LDK):
                uops.append(("ldr_stub", f0, op == E.C_LDRH, op == E.C_LDK, (w >> 12) if op == E.C_LDK else -1))
                pc += 1
            elif op == E.C_FILL:
                uops.append(("fill", f0, int(c[pc + 1])))
                pc += 2
            elif op == E.C_CPD:
                uops.append(("cpd", f0, f1))
                pc += 1
            elif op in C_BIN_REV:
                uops.append(("op", C_BIN_REV[op], f0, f1, f2))
                pc += 1
            elif op == E.C_WHERE:
                uops.append(("op", "where", f0, f1, f2, f3))
                pc += 1
            elif op == E.C_MAD:
                uops.append(("op", "mad", f0, f1, f2, f3))
                pc += 1
            elif op == E.C_MSB:
                uops.append(("op", "msb", f0, f1, f2, f3))
                pc += 1
            elif op in C_UN_REV:
                uops.append(("un", C_UN_REV[op], f0))
                pc += 1
            elif op == E.C_FOLD:
                uops.append(("fold", f0))
                pc += 1
            else:
                raise ValueError(f"bad compute uop {op} at pc={pc}")
        else:
            if op == E.C_ST:
                stores.append((f0, ("L1", w >> 16, is_tile)))
                pc += 1
            elif op in (E.C_STW, E.C_STWH):
                stores.append((f0, ("STW_STUB", op == E.C_STWH)))
                pc += 1
            else:
                raise ValueError(f"bad compute store op {op} at pc={pc}")


def _parse_c_stream(c, paged=True):
    ops = []
    pc, n = 0, len(c)
    PW = E.PW
    while pc < n:
        w = int(c[pc])
        op = w & 0x7F
        if op == E.C_END:
            ops.append((pc, "END"))
            break
        elif op == E.C_JMP:
            if paged:
                # Real MIR jump is (C_JMP, 0) followed by pad; padding is C_JMP repeated to page end.
                if (pc % PW <= PW - 2) and int(c[pc + 1]) == 0:
                    ops.append((pc, "JMP"))
                pc = (pc // PW + 1) * PW
            else:
                ops.append((pc, "JMP"))
                pc += 2
        elif op == E.C_BRT:
            ops.append((pc, "BRT", int(c[pc + 1])))
            pc = (pc // PW + 1) * PW if paged else pc + 3
        elif op == E.C_PH:
            ops.append((pc, "PH", w >> 16))
            pc += 1
        elif op == E.C_TOK:
            pc += 1
        elif op == E.C_SIGB:
            ops.append((pc, "SIGB")); pc += 1
        elif op == E.C_SIGN:
            ops.append((pc, "SIGN")); pc += 1
        elif op == E.C_WAITB:
            ops.append((pc, "WAITB")); pc += 1
        elif op == E.C_WAITN:
            ops.append((pc, "WAITN")); pc += 1
        elif op == E.C_TBEG:
            bt = w >> 16
            pc += 1
            waitn_in_loop = False
            blks = []
            while True:
                w2 = int(c[pc])
                op2 = w2 & 0x7F
                if op2 == E.C_TEND:
                    pc += 1
                    break
                elif op2 == E.C_JMP and paged:
                    pc = (pc // PW + 1) * PW
                elif op2 == E.C_TOK:
                    pc += 1
                elif op2 == E.C_WAITN:
                    waitn_in_loop = True
                    pc += 1
                elif op2 == E.C_ACQ:
                    uops, stores, pc = _parse_c_block(c, pc, paged=paged)
                    blks.append((uops, stores))
                else:
                    raise ValueError(f"unexpected op {op2} inside C_TBEG at pc={pc}")
            if waitn_in_loop:
                ops.append((pc, "WAITN"))
            ops.append((pc, "BLK_RUN", bt, blks))
        elif op == E.C_ACQ:
            uops, stores, pc = _parse_c_block(c, pc, paged=paged)
            ops.append((pc, "BLK_RUN", 1, [(uops, stores)]))
        else:
            raise ValueError(f"unexpected C op {op} at pc={pc}")
    return ops


def _merge_streams(c_ops, b_ops, n_ops):
    """Lockstep linear merge of C, B, N streams back into MIR order, matching N_STREAM and B_SINK to BLK_RUNs."""
    # Compute regions on B stream using backward jumps (exact match with census.region_of)
    b_loops = sorted({(op[2] if op[1] == "JMP" else op[3], op[0])
                      for op in b_ops if op[1] in ("JMP", "BRT")
                      and (op[2] if op[1] == "JMP" else op[3]) < op[0]})
    outer = max(b_loops, key=lambda l: l[1] - l[0]) if b_loops else None
    inner = [l for l in b_loops if l != outer]
    tops = [l for l in inner if not any(o != l and o[0] <= l[0] and l[1] <= o[1] for o in inner)]

    def b_region(bpc):
        k = next((i for i, l in enumerate(tops) if l[0] <= bpc <= l[1]), None)
        if k is not None:
            return f"solve{k}"
        if outer and outer[0] <= bpc <= outer[1]:
            return "outer"
        return "top"

    ic = ib = in_ = 0
    q = Counter()
    ph = "top"
    events = []
    idx = 0

    def cur_reg():
        return b_region(b_ops[ib][0]) if ib < len(b_ops) else "top"

    while ic < len(c_ops) or ib < len(b_ops) or in_ < len(n_ops):
        c_cur = c_ops[ic] if ic < len(c_ops) else None
        b_cur = b_ops[ib] if ib < len(b_ops) else None
        n_cur = n_ops[in_] if in_ < len(n_ops) else None

        # 1. End
        if c_cur and c_cur[1] == "END" and b_cur and b_cur[1] == "END" and n_cur and n_cur[1] == "END":
            ic += 1; ib += 1; in_ += 1
            break

        # 2. Phase markers (C and B together)
        if c_cur and c_cur[1] == "PH" and b_cur and b_cur[1] == "PH":
            assert c_cur[2] == b_cur[2], f"PH mismatch {c_cur} vs {b_cur}"
            ph = PHASES[c_cur[2]]
            ic += 1; ib += 1
            continue

        # 3. Control flow (C, B, N together)
        if c_cur and c_cur[1] == "JMP" and b_cur and b_cur[1] == "JMP" and n_cur and n_cur[1] == "JMP":
            ic += 1; ib += 1; in_ += 1
            continue
        if c_cur and c_cur[1] == "BRT" and b_cur and b_cur[1] == "BRT" and n_cur and n_cur[1] == "BRT":
            assert c_cur[2] == b_cur[2] == n_cur[2], f"BRT flag mismatch {c_cur} {b_cur} {n_cur}"
            events.append((idx, b_region(b_cur[0]), ph, ("branch", c_cur[2])))
            idx += 1
            ic += 1; ib += 1; in_ += 1
            continue

        # 4. Halo (B and N together)
        if b_cur and b_cur[1] == "HALO" and n_cur and n_cur[1] == "HALO":
            assert b_cur[2] == n_cur[2] and b_cur[3] == n_cur[3]
            events.append((idx, b_region(b_cur[0]), ph, ("halo", b_cur[2], b_cur[3])))
            idx += 1
            ib += 1; in_ += 1
            continue

        # 5. Non-blocking single-agent signals & side-effect ops
        progressed = False
        # B non-blocking ops (before matching SINK)
        if b_cur:
            bop = b_cur[1]
            if bop == "SIGC":
                q["bc"] += 1; ib += 1; progressed = True
            elif bop == "SIGN":
                q["bn"] += 1; ib += 1; progressed = True
            elif bop == "WAITC" and q["cb"] > 0:
                q["cb"] -= 1; ib += 1; progressed = True
            elif bop == "WAITN" and q["nb"] > 0:
                q["nb"] -= 1; ib += 1; progressed = True
            elif bop == "FETCH":
                events.append((idx, b_region(b_cur[0]), ph, ("fetch", b_cur[2], b_cur[3], b_cur[4])))
                idx += 1; ib += 1; progressed = True
            elif bop == "RED":
                events.append((idx, b_region(b_cur[0]), ph, ("red", b_cur[2])))
                idx += 1; ib += 1; progressed = True
            elif bop == "OUT":
                events.append((idx, b_region(b_cur[0]), ph, ("out", b_cur[3], b_cur[2])))
                idx += 1; ib += 1; progressed = True
            elif bop == "BGATHER":
                events.append((idx, b_region(b_cur[0]), ph, ("gather", b_cur[2], b_cur[3], b_cur[4], 3)))
                idx += 1; ib += 1; progressed = True
            elif bop == "SCAL":
                events.append((idx, b_region(b_cur[0]), ph, ("bscal", b_cur[2])))
                idx += 1; ib += 1; progressed = True
        if progressed:
            continue

        # N non-blocking ops (before matching STREAM)
        if n_cur:
            nop = n_cur[1]
            if nop == "SIGC":
                q["nc"] += 1; in_ += 1; progressed = True
            elif nop == "SIGB":
                q["nb"] += 1; in_ += 1; progressed = True
            elif nop == "WAITC" and q["cn"] > 0:
                q["cn"] -= 1; in_ += 1; progressed = True
            elif nop == "WAITB" and q["bn"] > 0:
                q["bn"] -= 1; in_ += 1; progressed = True
            elif nop == "GATHER":
                events.append((idx, cur_reg(), ph, ("gather", n_cur[2], n_cur[3], n_cur[4], n_cur[5])))
                idx += 1; in_ += 1; progressed = True
        if progressed:
            continue

        # C non-blocking syncs
        if c_cur:
            cop = c_cur[1]
            if cop == "SIGB":
                q["cb"] += 1; ic += 1; progressed = True
            elif cop == "SIGN":
                q["cn"] += 1; ic += 1; progressed = True
            elif cop == "WAITB" and q["bc"] > 0:
                q["bc"] -= 1; ic += 1; progressed = True
            elif cop == "WAITN" and q["nc"] > 0:
                q["nc"] -= 1; ic += 1; progressed = True
        if progressed:
            continue

        # 6. BLK_RUN matched with optional N_STREAM and B_SINK
        if c_cur and c_cur[1] == "BLK_RUN":
            bt, blks = c_cur[2], c_cur[3]
            n_ldrs = sum(1 for uops, _ in blks for u in uops if u[0] == "ldr_stub")
            n_stws = sum(1 for _, sts in blks for _, dst in sts if dst[0] == "STW_STUB")
            if n_ldrs > 0 and (not n_cur or n_cur[1] != "STREAM"):
                raise RuntimeError(f"BLK_RUN needs {n_ldrs} STREAM entries, N at {n_cur}")
            if n_stws > 0 and (not b_cur or b_cur[1] != "SINK"):
                raise RuntimeError(f"BLK_RUN needs {n_stws} SINK entries, B at {b_cur}")
            stream_ents = []
            if n_ldrs > 0:
                stn, stream_ents = n_cur[2], n_cur[3]
                assert len(stream_ents) == n_ldrs, f"STREAM len {len(stream_ents)} != ldrs {n_ldrs}"
                assert stn == bt or (bt == 1 and stn >= 1), f"STREAM tn {stn} != bt {bt}"
                in_ += 1
            sink_ents = []
            reg_here = cur_reg()
            if n_stws > 0:
                btn, sink_ents = b_cur[2], b_cur[3]
                reg_here = b_region(b_cur[0])
                assert len(sink_ents) == n_stws, f"SINK len {len(sink_ents)} != stws {n_stws}"
                assert btn == bt or (bt == 1 and btn >= 1), f"SINK tn {btn} != bt {bt}"
                ib += 1
            it_ldr = iter(stream_ents)
            it_stw = iter(sink_ents)
            new_blks = []
            for uops, sts in blks:
                nu = []
                for u in uops:
                    if u[0] == "ldr_stub":
                        src_desc, shift, mid, parts, narrow, kfold = next(it_ldr)
                        nu.append(("ldr", u[1], src_desc, shift, mid, parts, narrow, kfold))
                    else:
                        nu.append(u)
                ns = []
                for s, dst in sts:
                    if dst[0] == "STW_STUB":
                        sdesc, _ = next(it_stw)
                        assert sdesc[0] == "HOME"
                        ns.append((s, ("HOME", sdesc[1], True)))
                    else:
                        ns.append((s, dst))
                new_blks.append((nu, ns))
            events.append((idx, reg_here, ph, ("blk_run", bt, new_blks)))
            idx += 1
            ic += 1
            continue

        raise RuntimeError(f"Stream merge stuck at ic={ic} ({c_cur}), ib={ib} ({b_cur}), in={in_} ({n_cur}), q={dict(q)}")

    assert ic == len(c_ops) and ib == len(b_ops) and in_ == len(n_ops), (ic, len(c_ops), ib, len(b_ops), in_, len(n_ops))
    assert all(v == 0 for v in q.values()), f"unconsumed sync tokens: {dict(q)}"
    return events


def _tabstream_map_counts(L, nparts_by_mid, T, wide=False):
    """Per-tile (loc, halo, zero, is_dense) counts for every map in core's streamed L1 table L."""
    nmaps = int(L[0])
    out = {}
    for mid, nparts in nparts_by_mid.items():
        assert mid < nmaps, f"mid {mid} >= nmaps {nmaps}"
        w = int(L[1 + mid])
        q, tm = w & 0xFFFFF, w >> 20
        is_dense = bool(int(L[q + 1]) & DENSE_BIT)
        nsec = 2 * nparts + 1
        nh = nsec * tm + 1
        if wide:
            H = np.asarray(L[q + 2:q + 2 + nh], np.int64)
        else:
            words = np.asarray(L[q + 2:q + 2 + (nh + 1) // 2], np.int64)
            H = np.stack([words & 0xFFFF, words >> 16], 1).ravel()[:nh]
        loc = np.zeros(T, np.int64)
        halo = np.zeros(T, np.int64)
        zero = np.zeros(T, np.int64)
        for p in range(nparts):
            a_l = H[(2 * p) * tm:(2 * p + 1) * tm]
            b_l = H[(2 * p) * tm + 1:(2 * p + 1) * tm + 1]
            cnt_l = b_l - a_l
            if is_dense and p == 0:
                # Dense table stores 512 uint32 words = 1024 uint16 source indices per tile
                cnt_l = cnt_l * 2
            loc[:tm] += cnt_l
            a_h = H[(2 * p + 1) * tm:(2 * p + 2) * tm]
            b_h = H[(2 * p + 1) * tm + 1:(2 * p + 2) * tm + 1]
            halo[:tm] += b_h - a_h
        a_z = H[(2 * nparts) * tm:(2 * nparts + 1) * tm]
        b_z = H[(2 * nparts) * tm + 1:(2 * nparts + 1) * tm + 1]
        zero[:tm] += b_z - a_z
        out[mid] = (loc, halo, zero, is_dense)
    return out


def load_from_image(img_dir, core):
    d = Path(img_dir)
    z = np.load(d / "shared.npz")
    meta = json.loads(bytes(z["meta"]).decode())
    sc = meta["scalars"]
    P = int(sc["P"])
    T = int(sc["T"])
    K = int(sc["K"])
    chip = core // P
    lcore = core % P
    assert 0 <= chip < K, f"core {core} out of range for K={K}, P={P}"
    flags = meta.get("flags", {})
    paged = bool(flags.get("page", 0))
    wide = T > 64

    c_ops = _parse_c_stream(z["c"], paged=paged)
    b_ops = _parse_b_stream(z["b"], T)
    n_ops = _parse_n_stream(z["n"], T, wide=wide)
    events = _merge_streams(c_ops, b_ops, n_ops)

    # Collect nparts per referenced map id
    nparts_by_mid = {}
    for _, _, _, ev in events:
        if ev[0] == "blk_run":
            for uops, _ in ev[2]:
                for u in uops:
                    if u[0] == "ldr" and u[4] is not None:
                        nparts_by_mid[u[4]] = len(u[5])
        elif ev[0] == "gather":
            nparts_by_mid[ev[2]] = len(ev[3])

    ch_dir = d / f"chip{chip}"
    consts_mm = np.load(ch_dir / "consts.npy", mmap_mode="r")
    consts_u32 = np.ascontiguousarray(consts_mm[lcore]).view(np.uint32)
    const_uni = page_uniform_u32(consts_u32)

    arena_mm = np.load(ch_dir / "arena.npy", mmap_mode="r")
    arena_u32 = np.ascontiguousarray(arena_mm[lcore]).view(np.uint32)
    arena_uni = page_uniform_u32(arena_u32)

    home_mm = np.load(ch_dir / "home.npy", mmap_mode="r")
    home_u32 = np.ascontiguousarray(home_mm[lcore]).view(np.uint32)
    home_uni = page_uniform_u32(home_u32)

    init_ro = {}
    for ti, b in enumerate(arena_uni.tolist()):
        if b >= 0:
            init_ro[("L1", ti)] = b
    for hi, b in enumerate(home_uni.tolist()):
        if b >= 0:
            init_ro[("HOME", hi)] = b

    cc = np.load(ch_dir / "core@cells.npy", mmap_mode="r")
    cells = int(np.sum(cc == lcore))

    tab_mm = np.load(ch_dir / "tab.npy", mmap_mode="r")
    L = np.array(tab_mm[lcore], dtype=np.uint32)
    assert sc.get("tab_stream", False), "only tab_stream=True images supported"
    map_counts = _tabstream_map_counts(L, nparts_by_mid, T, wide=wide)

    state_locs = set()
    for _, _, regs in meta["state"]:
        for _, phys, sz in regs:
            if phys is None:
                continue
            sp = "HOME" if phys >= DBASE else "L1"
            base = phys - DBASE if phys >= DBASE else phys
            for t in range(sz):
                state_locs.add((sp, base + t))

    return dict(
        name=d.name,
        core=core,
        T=T,
        cells=cells,
        cpages=int(sc["cpages"]),
        const_uni=const_uni,
        init_ro=init_ro,
        state_locs=state_locs,
        map_counts=map_counts,
        events=events,
        flags=flags,
    )


# ==================================================================================================
# Core Engine: Inter-Instruction Per-Tile Propagation & Backward Liveness
# ==================================================================================================

def _locs_written(events):
    """Set of (space, tile_idx) written anywhere in events (conservative across all tiles 0..bt-1)."""
    wr = set()
    for _, _, _, ev in events:
        k = ev[0]
        if k == "blk_run":
            bt, blks = ev[1], ev[2]
            for _, stores in blks:
                for _, (sp, base, is_vec) in stores:
                    if is_vec:
                        for t in range(bt):
                            wr.add((sp, base + t))
                    else:
                        wr.add((sp, base))
        elif k == "gather":
            # Conservatively mark all T tiles if needed; refined using map_counts below
            wr.add(("GATHER_L1", ev[1]))
        elif k == "fetch":
            for t in range(ev[3]):
                wr.add(("L1", ev[1] + t))
        elif k == "red":
            for _, fold_l1 in ev[1]:
                wr.add(("L1", fold_l1))
    return wr


def _desc_locs(desc, T):
    if desc is None:
        return []
    sp, base = desc
    if sp == "CONST":
        return []
    return [(sp, base + t) for t in range(T)]


def analyze_region(prog, care=True, use_ro_nonconst=True):
    """Run inter-instruction per-tile propagation across outer and each inner solver loop solve0..solveK.

    Returns per-(region, phase) and aggregate counts of total vs dead:
      - op_tiles (and by op name: mul, add, sub, div, where, mad, msb, max, min, un, fold)
      - gathered_vals (patched_vals on ldr + gather_vals on standalone gather)
      - stream_pages (all N_STREAM ldr pages: const + home + l1/zero; also broken out into nonconst_ldr and const_ldr)
      - const_pages (total, uniform_kfold, dead_total, dead_nonuniform_extra)
      - dram_stores (C_STW total, dead, and blocking reasons for zero/copy stores kept live)
      - tile_passes (blk_run tile-passes total vs completely dead)
    """
    T = prog["T"]
    const_uni = prog["const_uni"]
    state_locs = set(prog["state_locs"])
    map_counts = prog["map_counts"]
    events = prog["events"]

    # Identify all locations written in outer or any solve loop
    in_body = [e for e in events if e[1] != "top"]
    written_all = set()
    written_by_reg = defaultdict(set)
    for _, reg, _, ev in in_body:
        k = ev[0]
        if k == "blk_run":
            bt, blks = ev[1], ev[2]
            for _, stores in blks:
                for _, (sp, base, is_vec) in stores:
                    for t in range(bt if is_vec else 1):
                        written_all.add((sp, base + t))
                        written_by_reg[reg].add((sp, base + t))
        elif k == "gather":
            dst_l1, mid = ev[1], ev[2]
            for t in range(T):
                written_all.add(("L1", dst_l1 + t))
                written_by_reg[reg].add(("L1", dst_l1 + t))
        elif k == "fetch":
            for t in range(ev[3]):
                written_all.add(("L1", ev[1] + t))
                written_by_reg[reg].add(("L1", ev[1] + t))
        elif k == "red":
            for _, fold_l1 in ev[1]:
                written_all.add(("L1", fold_l1))
                written_by_reg[reg].add(("L1", fold_l1))

    # Read-only pre-initialized L1 / HOME locations (like cell_mask = 1.0 on tiles 0..T-2)
    ro_init = {}
    if use_ro_nonconst:
        for loc, bits in prog["init_ro"].items():
            if loc not in written_all and loc not in state_locs:
                ro_init[loc] = bits

    # Node graph for backward liveness across the whole program (outer + each solve loop)
    # Each node: dict(cat, reg, ph, deps, weight, meta)
    nodes = []
    roots = []
    block_reasons = Counter()

    def add_node(cat, reg, ph, deps=(), weight=1, meta=None):
        nid = len(nodes)
        nodes.append((cat, reg, ph, [d for d in deps if d is not None], weight, meta))
        return nid

    # Value representation:
    #   ("K", bits_u32, is_scalar)      -- known uniform finite fp32
    #   ("V", vid, origin_list)         -- runtime-varying, origin_list = [(loc, ver, store_nid), ...] oldest first
    next_vid = [1]

    def fresh_v(origin=()):
        vid = next_vid[0]
        next_vid[0] += 1
        return ("V", vid, list(origin))

    # Memory state: loc -> (val, ver, def_nid)
    #   def_nid is None for initial/unknown/ro values, or the store/gather/fetch node id that produced it.
    mem = {}
    ver_counter = Counter()

    def mem_get(loc):
        if loc in mem:
            return mem[loc]
        if loc in ro_init:
            return (("K", ro_init[loc], False), 0, None)
        v = fresh_v([(loc, 0, None)])
        mem[loc] = (v, 0, None)
        return mem[loc]

    def mem_set(loc, val, def_nid):
        ver_counter[loc] += 1
        ver = ver_counter[loc]
        if val[0] == "V":
            # Append this home to origin_list (oldest first, no dup)
            orig = list(val[2]) + [(loc, ver, def_nid)]
            val = ("V", val[1], orig)
        mem[loc] = (val, ver, def_nid)
        return val, ver

    def can_take_origin(val, reader_tier, cur_loc):
        """Find earliest equivalent home in val's origin list that is un-clobbered and tier-compatible.
        reader_tier: 'L1' (C_LD) or 'STREAM' (C_LDR).
        Returns (chosen_loc, chosen_def_nid, bypassed, block_reason_if_not_bypassed)."""
        origs = val[2]
        if not origs:
            return cur_loc, None, False, None
        # Last entry is cur_loc itself (if val came from cur_loc)
        cur_def = mem[cur_loc][2] if cur_loc in mem else None
        if len(origs) <= 1:
            return cur_loc, cur_def, False, None
        reason = None
        for oloc, over, odef in origs[:-1]:
            # Check if oloc still holds version over
            if oloc[0] == "CONST":
                # Constant is never clobbered; compatible with STREAM only (not L1 C_LD without adding DRAM read)
                if reader_tier == "STREAM":
                    return oloc, odef, True, None
                reason = reason or "tier_l1_from_const"
                continue
            cur_m = mem.get(oloc)
            still_alive = (cur_m is not None and cur_m[1] == over)
            if not still_alive:
                reason = reason or "source_clobbered"
                continue
            if reader_tier == "L1" and oloc[0] != "L1":
                reason = reason or "tier_l1_from_dram"
                continue
            return oloc, odef, True, None
        return cur_loc, cur_def, False, reason

    def map_tile_counts(mid, t):
        if mid not in map_counts:
            return 0, 0, 0, False
        lc, hc, zc, is_d = map_counts[mid]
        if t >= len(lc):
            return 0, 0, 0, is_d
        return int(lc[t]), int(hc[t]), int(zc[t]), is_d

    # Process events in order, handling solver loops carefully:
    # For each solve loop solveK:
    #   Before entering solveK, snapshot `mem` (loop-invariant values produced by outer!).
    #   Any location written inside solveK is loop-carried across iterations of solveK (unless it is only
    #   written before being read in every iteration, in which case its entry value is never used!).
    #   Even better: initialize every location in `written_by_reg[solveK]` to UNKNOWN at the header of solveK!
    #   Wait: check if any tile written in solveK is ALWAYS written with the same constant K? Let's do a 2-pass
    #   fixed-point check: pass 1 marks any read-before-write location in `written_by_reg[solveK]` as UNKNOWN;
    #   wait — setting ALL `written_by_reg[solveK]` to UNKNOWN before solveK is 100% sound (and any temp written
    #   BEFORE read inside solveK gets overwritten with its exact value immediately anyway!).
    #   And AFTER exiting solveK back into outer, any location written in solveK is UNKNOWN (unless we know
    #   for sure solveK ran >= 1 it with fixed constant — treating it as UNKNOWN in outer is 100% sound!).

    # Group events into segments by contiguous region transitions
    seen_solve = set()
    active_solve = None

    # Track per-region, per-phase totals & node mappings
    # Also track standalone gather tile nodes so we know which tiles of a gather are live!
    stats = defaultdict(Counter)
    pass_records = []  # (reg, ph, bt_pass_store_nodes)

    for ev_i, reg, ph, ev in events:
        if reg == "top":
            continue

        # Handle region transition into / out of solveK
        if reg != active_solve:
            if active_solve is not None and active_solve.startswith("solve"):
                # Exiting solveK: any location written in solveK may carry arbitrary value back to outer
                # (also mark loop-carried live-out of solveK!)
                for loc in written_by_reg[active_solve]:
                    if loc in mem and mem[loc][2] is not None:
                        # Loop-carried store at end of solveK: keep live if read in solveK or outer
                        roots.append(mem[loc][2])
                        if mem[loc][0][0] == "K" or (mem[loc][0][0] == "V" and len(mem[loc][0][2]) > 1):
                            block_reasons["loop_carried_solve"] += 1
                    mem_set(loc, fresh_v(), None)
            if reg.startswith("solve") and reg not in seen_solve:
                seen_solve.add(reg)
                # Entering solveK: locations written in solveK could have changed on previous inner iterations
                for loc in written_by_reg[reg]:
                    # Keep outer's store into loc live (it initializes the solver state!)
                    if loc in mem and mem[loc][2] is not None:
                        roots.append(mem[loc][2])
                        if mem[loc][0][0] == "K" or (mem[loc][0][0] == "V" and len(mem[loc][0][2]) > 1):
                            block_reasons["solve_entry_state"] += 1
                    mem_set(loc, fresh_v(), None)
            active_solve = reg

        k = ev[0]
        st = stats[(reg, ph)]

        if k == "fetch":
            dst_l1, coff, ntiles = ev[1], ev[2], ev[3]
            for t in range(ntiles):
                loc = ("L1", dst_l1 + t)
                st["fetch_pages"] += 1
                g = coff + t
                b = int(const_uni[g]) if g < len(const_uni) else -1
                if b >= 0:
                    nid = add_node("fetch_const_uni", reg, ph, (), 1, loc)
                    mem_set(loc, ("K", b, False), nid)
                else:
                    nid = add_node("fetch_const", reg, ph, (), 1, loc)
                    mem_set(loc, fresh_v([(("CONST", g), 0, None)]), nid)

        elif k == "gather":
            dst_l1, mid, parts, mode = ev[1], ev[2], ev[3], ev[4]
            # A standalone gather writes L1 tiles 0..tm-1 (all tiles where map has entries, or all T tiles).
            # Check per-tile entry counts!
            for t in range(T):
                nl, nh, nz, is_d = map_tile_counts(mid, t)
                gvals = nl + nh + nz
                if gvals == 0:
                    continue
                st["gather_tiles"] += 1
                st["gather_vals"] += gvals
                st["gathered_vals"] += gvals
                # Gather reads arbitrary lanes from each part's source -> mark all tiles of part sources as non-plain deps
                loc = ("L1", dst_l1 + t)
                deps = []
                # A non-dense gather that writes fewer than LANES entries preserves untouched lanes of A[dst]
                # (e.g. a split gather where a preceding `lds` block filled the window lanes of dst!).
                full_overwrite = is_d or (nl + nh + nz >= LANES)
                prior_zero = True
                if not full_overwrite:
                    pval, _, pdef = mem_get(loc)
                    if pdef is not None:
                        deps.append(pdef)
                    if not (pval[0] == "K" and is_zero_u32(pval[1])):
                        prior_zero = False
                if nl == 0 and nh == 0:
                    # Only zero-fills on this tile (e.g. boundary gather with no boundary faces in tile t)
                    all_parts_zero = prior_zero
                else:
                    all_parts_zero = prior_zero and (nh == 0)
                    for pdesc, _ in parts:
                        if pdesc is None:
                            continue
                        psp, pbase = pdesc
                        if psp == "CONST":
                            all_parts_zero = False
                        else:
                            for st_t in range(T):
                                sloc = (psp, pbase + st_t)
                                sval, _, sdef = mem_get(sloc)
                                if sdef is not None:
                                    deps.append(sdef)
                                    if sval[0] == "K" or (sval[0] == "V" and len(sval[2]) > 1):
                                        block_reasons["non_plain_gather_reader"] += 1
                                if not (sval[0] == "K" and is_zero_u32(sval[1])):
                                    all_parts_zero = False
                gnid = add_node("gather_tile", reg, ph, deps, gvals, (mid, t))
                if all_parts_zero:
                    mem_set(loc, ("K", ZERO_POS, False), gnid)
                else:
                    mem_set(loc, fresh_v(), gnid)

        elif k == "halo":
            _, src_desc = ev[1], ev[2]
            if src_desc is not None and src_desc[0] != "CONST":
                psp, pbase = src_desc
                for t in range(T):
                    sloc = (psp, pbase + t)
                    sval, _, sdef = mem_get(sloc)
                    if sdef is not None:
                        roots.append(sdef)
                        if sval[0] == "K" or (sval[0] == "V" and len(sval[2]) > 1):
                            block_reasons["non_plain_halo_reader"] += 1

        elif k == "red":
            for part_l1, fold_l1 in ev[1]:
                sloc = ("L1", part_l1)
                sval, _, sdef = mem_get(sloc)
                if sdef is not None:
                    roots.append(sdef)
                mem_set(("L1", fold_l1), fresh_v(), None)

        elif k == "out":
            sloc = ("L1", ev[2])
            sval, _, sdef = mem_get(sloc)
            if sdef is not None:
                roots.append(sdef)

        elif k == "branch":
            sloc = ("L1", ev[1])
            sval, _, sdef = mem_get(sloc)
            if sdef is not None:
                roots.append(sdef)

        elif k == "bscal":
            # Scalar BRISC block; rare/not in control, conservatively mark any L1 tiles live
            pass

        elif k == "blk_run":
            bt, blks = ev[1], ev[2]
            for t in range(bt):
                st["passes"] += 1
                pass_store_nodes = []
                for uops, stores in blks:
                    # Dest slots S[slot] = (val, producer_nid)
                    # Note: if val is ("K", ...), producer_nid is None (free immediate / fill, no memory dep!).
                    S = {}
                    for u in uops:
                        uk = u[0]
                        if uk == "ld":
                            dst_s, (sp, base, is_vec) = u[1], u[2]
                            loc = (sp, base + (t if is_vec else 0))
                            val, _, def_nid = mem_get(loc)
                            st["ld_l1"] += 1
                            if val[0] == "K":
                                # Known uniform constant -> no L1 load needed!
                                n = add_node("ld_l1_folded", reg, ph, (), 1)
                                S[dst_s] = (val, None)
                            else:
                                _, cdef, bypassed, breason = can_take_origin(val, "L1", loc)
                                if not bypassed and breason:
                                    block_reasons[breason] += 1
                                n = add_node("ld_l1", reg, ph, [cdef], 1)
                                S[dst_s] = (val, n)

                        elif uk == "rvf":
                            dst_s, (sp, base, _) = u[1], u[2]
                            loc = (sp, base)
                            val, _, def_nid = mem_get(loc)
                            st["ld_l1"] += 1
                            if val[0] == "K":
                                S[dst_s] = (("K", val[1], False), None)
                            else:
                                n = add_node("ld_l1", reg, ph, [def_nid], 1)
                                S[dst_s] = (fresh_v(), n)

                        elif uk == "lds":
                            dst_s, (sp, base, is_vec), shift = u[1], u[2], u[3]
                            st["lds_l1"] += 1
                            # Reads lanes [t*1024 + shift, (t+1)*1024 + shift)
                            lo_lane = t * LANES + shift
                            hi_lane = (t + 1) * LANES + shift - 1
                            t0, t1 = lo_lane // LANES, hi_lane // LANES
                            deps = []
                            same_k = True
                            kval = None
                            if t0 < 0 or t1 >= T:
                                same_k = False
                            for tt in range(max(0, t0), min(T, t1 + 1)):
                                loc = (sp, base + (tt if is_vec else 0))
                                val, _, def_nid = mem_get(loc)
                                if def_nid is not None:
                                    deps.append(def_nid)
                                if val[0] == "K" and not val[2]:
                                    if kval is None:
                                        kval = val[1]
                                    elif kval != val[1]:
                                        same_k = False
                                else:
                                    same_k = False
                            if same_k and kval is not None:
                                S[dst_s] = (("K", kval, False), None)
                            else:
                                for tt in range(max(0, t0), min(T, t1 + 1)):
                                    loc = (sp, base + (tt if is_vec else 0))
                                    val, _, def_nid = mem_get(loc)
                                    if def_nid is not None and (val[0] == "K" or (val[0] == "V" and len(val[2]) > 1)):
                                        block_reasons["non_plain_lds_reader"] += 1
                                n = add_node("lds", reg, ph, deps, 1)
                                S[dst_s] = (fresh_v(), n)

                        elif uk == "ldr":
                            dst_s, src_desc, shift, mid, parts, is_narrow, _ = u[1:]
                            plain = (shift == 0 and mid is None and not parts)
                            st["stream_pages"] += 1

                            # Gather/patch counts for this tile
                            nl = nh = nz = 0
                            is_d = False
                            if mid is not None:
                                nl, nh, nz, is_d = map_tile_counts(mid, t)
                                gvals = nl + nh + nz
                                st["patched_vals"] += gvals
                                st["gathered_vals"] += gvals
                                if gvals > 0:
                                    st["patched_pages"] += 1
                            else:
                                gvals = 0

                            if src_desc is None:
                                # Zero window (+ optional patch)
                                st["ldr_zero_pages"] += 1
                                if gvals == 0 or (nl == 0 and nh == 0):
                                    # Zero window with zero patch entries (or only zero-fills!) -> +0.0!
                                    # The stream page read (and any zero-fill patches) is dead!
                                    add_node("ldr_zero_dead", reg, ph, (), 1, gvals)
                                    S[dst_s] = (("K", ZERO_POS, False), None)
                                else:
                                    # Check if all patched parts are non-zero or zero
                                    pdeps = []
                                    all_pzero = (nh == 0)
                                    for pdesc, _ in parts:
                                        if pdesc is None:
                                            continue
                                        psp, pbase = pdesc
                                        if psp == "CONST":
                                            all_pzero = False
                                        else:
                                            for st_t in range(T):
                                                sloc = (psp, pbase + st_t)
                                                sval, _, sdef = mem_get(sloc)
                                                if sdef is not None:
                                                    pdeps.append(sdef)
                                                if not (sval[0] == "K" and is_zero_u32(sval[1])):
                                                    all_pzero = False
                                    if all_pzero:
                                        add_node("ldr_zero_dead", reg, ph, (), 1, gvals)
                                        S[dst_s] = (("K", ZERO_POS, False), None)
                                    else:
                                        n = add_node("ldr_zero", reg, ph, pdeps, 1, gvals)
                                        S[dst_s] = (fresh_v(), n)

                            elif src_desc[0] == "CONST":
                                coff = src_desc[1]
                                g = coff + t
                                st["const_pages"] += 1
                                b = int(const_uni[g]) if g < len(const_uni) else -1
                                if plain and b >= 0:
                                    st["const_pages_uniform"] += 1
                                    if is_zero_u32(b):
                                        st["const_pages_zero"] += 1
                                    # Uniform constant page: folded (never streamed; counted as dead const read!)
                                    add_node("ldr_const_uni", reg, ph, (), 1, 0)
                                    S[dst_s] = (("K", b, False), None)
                                elif plain:
                                    n = add_node("ldr_const", reg, ph, (), 1, 0)
                                    S[dst_s] = (fresh_v([(("CONST", g), 0, None)]), n)
                                else:
                                    # Shifted or patched constant read
                                    pdeps = []
                                    for pdesc, _ in parts:
                                        if pdesc is not None and pdesc[0] != "CONST":
                                            for st_t in range(T):
                                                _, _, sdef = mem_get((pdesc[0], pdesc[1] + st_t))
                                                if sdef is not None:
                                                    pdeps.append(sdef)
                                    n = add_node("ldr_const_pat", reg, ph, pdeps, 1, gvals)
                                    S[dst_s] = (fresh_v(), n)

                            else:
                                # HOME or L1 stream source (+ optional shift/patch)
                                ssp, sbase = src_desc
                                st["ldr_home_pages" if ssp == "HOME" else "ldr_l1_pages"] += 1
                                if plain:
                                    loc = (ssp, sbase + t)
                                    val, _, def_nid = mem_get(loc)
                                    if val[0] == "K":
                                        # Known uniform tile (propagated zero/const or read-only cell_mask=1.0!)
                                        # Stream read replaced by immediate fill -> stream page read is DEAD!
                                        add_node("ldr_home_folded", reg, ph, (), 1, 0)
                                        S[dst_s] = (val, None)
                                    else:
                                        cloc, cdef, bypassed, breason = can_take_origin(val, "STREAM", loc)
                                        if not bypassed and breason:
                                            block_reasons[breason] += 1
                                        # Keep category as ldr_home (it is a non-const stream read in the original IR;
                                        # bypassing only rewrites its address to cloc so the copy store can die).
                                        n = add_node("ldr_home", reg, ph, [cdef], 1, 0)
                                        S[dst_s] = (val, n)
                                else:
                                    # Shifted and/or patched stream from HOME/L1
                                    lo_lane = t * LANES + shift
                                    hi_lane = (t + 1) * LANES + shift - 1
                                    t0, t1 = lo_lane // LANES, hi_lane // LANES
                                    deps = []
                                    win_zero = (t0 >= 0 and t1 < T)
                                    for tt in range(max(0, t0), min(T, t1 + 1)):
                                        sval, _, sdef = mem_get((ssp, sbase + tt))
                                        if sdef is not None:
                                            deps.append(sdef)
                                        if not (sval[0] == "K" and is_zero_u32(sval[1])):
                                            win_zero = False
                                    all_pzero = (nh == 0)
                                    for pdesc, _ in parts:
                                        if pdesc is None:
                                            continue
                                        psp, pbase = pdesc
                                        if psp == "CONST":
                                            all_pzero = False
                                        else:
                                            for st_t in range(T):
                                                sval, _, sdef = mem_get((psp, pbase + st_t))
                                                if sdef is not None:
                                                    deps.append(sdef)
                                                if not (sval[0] == "K" and is_zero_u32(sval[1])):
                                                    all_pzero = False
                                    if win_zero and (nl == 0 and nh == 0 or all_pzero):
                                        add_node("ldr_home_folded", reg, ph, (), 1, gvals)
                                        S[dst_s] = (("K", ZERO_POS, False), None)
                                    else:
                                        n = add_node("ldr_home_pat", reg, ph, deps, 1, gvals)
                                        S[dst_s] = (fresh_v(), n)

                        elif uk == "fill":
                            dst_s, bits = u[1], u[2]
                            st["fills"] += 1
                            if is_nan_u32(bits):
                                n = add_node("fill", reg, ph, (), 1)
                                S[dst_s] = (fresh_v(), n)
                            else:
                                S[dst_s] = (("K", bits, False), None)

                        elif uk == "cpd":
                            src_s, dst_s = u[1], u[2]
                            st["cpds"] += 1
                            v, pn = S[src_s]
                            if v[0] == "K":
                                S[dst_s] = (v, None)
                            else:
                                n = add_node("cpd", reg, ph, [pn], 1)
                                S[dst_s] = (v, n)

                        elif uk == "un":
                            name, slot = u[1], u[2]
                            st["op_tiles"] += 1
                            st["op:" + name] += 1
                            v, pn = S[slot]
                            if v[0] == "K":
                                res_b = eval_un_u32(name, v[1])
                                if res_b is not None:
                                    st["op_dead_const"] += 1
                                    S[slot] = (("K", res_b, v[2]), None)
                                    continue
                            n = add_node("op", reg, ph, [pn], 1, name)
                            S[slot] = (fresh_v(), n)

                        elif uk == "fold":
                            slot = u[1]
                            st["op_tiles"] += 1
                            st["op:fold"] += 1
                            v, pn = S[slot]
                            if v[0] == "K" and not v[2]:
                                res_b = eval_fold_u32(v[1])
                                if res_b is not None:
                                    st["op_dead_const"] += 1
                                    S[slot] = (("K", res_b, True), None)
                                    continue
                            n = add_node("op", reg, ph, [pn], 1, "fold")
                            S[slot] = (fresh_v(), n)

                        elif uk == "op":
                            name, dst_s = u[1], u[2]
                            args = [S[a] for a in u[3:]]
                            vals = [a[0] for a in args]
                            st["op_tiles"] += 1
                            st["op:" + name] += 1

                            folded = None  # (val, prod_nid, reason)
                            if name == "where":
                                cv, av, bv = vals
                                if cv[0] == "K":
                                    chosen = args[1] if not is_zero_u32(cv[1]) else args[2]
                                    folded = (chosen[0], chosen[1], "op_dead_ident_exact")
                                elif av[0] == "K" and bv[0] == "K":
                                    if av[1] == bv[1]:
                                        folded = (av, None, "op_dead_ident_exact")
                                    elif care and is_zero_u32(av[1]) and is_zero_u32(bv[1]):
                                        folded = (("K", ZERO_POS, av[2] and bv[2]), None, "op_dead_ident_care")
                                elif av[0] == "V" and bv[0] == "V" and av[1] == bv[1]:
                                    folded = (args[1][0], args[1][1], "op_dead_ident_exact")

                            elif all(v[0] == "K" for v in vals):
                                if name in ("mad", "msb"):
                                    res_b = eval_mad_u32(name, vals[0][1], vals[1][1], vals[2][1])
                                else:
                                    res_b = eval_bin_u32(name, vals[0][1], vals[1][1])
                                if res_b is not None:
                                    folded = (("K", res_b, any(v[2] for v in vals)), None, "op_dead_const")

                            elif name == "mul":
                                if vals[0][0] == "K" and vals[0][1] == ONE_U32:
                                    folded = (args[1][0], args[1][1], "op_dead_ident_exact")
                                elif vals[1][0] == "K" and vals[1][1] == ONE_U32:
                                    folded = (args[0][0], args[0][1], "op_dead_ident_exact")
                                elif care and ((vals[0][0] == "K" and is_zero_u32(vals[0][1])) or
                                               (vals[1][0] == "K" and is_zero_u32(vals[1][1]))):
                                    folded = (("K", ZERO_POS, False), None, "op_dead_zero_care")

                            elif name == "div":
                                if vals[1][0] == "K" and vals[1][1] == ONE_U32:
                                    folded = (args[0][0], args[0][1], "op_dead_ident_exact")
                                elif care and vals[0][0] == "K" and is_zero_u32(vals[0][1]):
                                    folded = (("K", ZERO_POS, False), None, "op_dead_zero_care")

                            elif name == "add":
                                for idx_i in (0, 1):
                                    if vals[idx_i][0] == "K" and is_zero_u32(vals[idx_i][1]):
                                        if care or vals[idx_i][1] == ZERO_NEG:
                                            r = "op_dead_ident_exact" if vals[idx_i][1] == ZERO_NEG else "op_dead_ident_care"
                                            folded = (args[1 - idx_i][0], args[1 - idx_i][1], r)
                                            break

                            elif name == "sub":
                                if vals[1][0] == "K" and is_zero_u32(vals[1][1]):
                                    if care or vals[1][1] == ZERO_POS:
                                        r = "op_dead_ident_exact" if vals[1][1] == ZERO_POS else "op_dead_ident_care"
                                        folded = (args[0][0], args[0][1], r)

                            elif name in ("mad", "msb"):
                                if care and ((vals[0][0] == "K" and is_zero_u32(vals[0][1])) or
                                             (vals[1][0] == "K" and is_zero_u32(vals[1][1]))):
                                    folded = (args[2][0], args[2][1], "op_dead_ident_care")
                                elif care and name == "mad" and vals[2][0] == "K" and is_zero_u32(vals[2][1]):
                                    if vals[0][0] == "K" and vals[0][1] == ONE_U32:
                                        folded = (args[1][0], args[1][1], "op_dead_ident_care")
                                    elif vals[1][0] == "K" and vals[1][1] == ONE_U32:
                                        folded = (args[0][0], args[0][1], "op_dead_ident_care")

                            elif name in ("max", "min"):
                                if vals[0][0] == "V" and vals[1][0] == "V" and vals[0][1] == vals[1][1]:
                                    folded = (args[0][0], args[0][1], "op_dead_ident_exact")

                            if folded is not None:
                                st[folded[2]] += 1
                                S[dst_s] = (folded[0], folded[1])
                            else:
                                n = add_node("op", reg, ph, [a[1] for a in args], 1, name)
                                S[dst_s] = (fresh_v(), n)
                        else:
                            raise NotImplementedError(uk)

                    # Process stores of this block
                    for s_slot, (sp, base, is_vec) in stores:
                        loc = (sp, base + (t if is_vec else 0))
                        val, pn = S[s_slot]
                        st["st_home" if sp == "HOME" else "st_l1"] += 1

                        # Check self-copy no-op store (e.g. acc = acc + 0 -> store acc to acc)
                        cur = mem.get(loc)
                        is_noop = False
                        if cur is not None:
                            cval = cur[0]
                            if val[0] == "V" and cval[0] == "V" and val[1] == cval[1]:
                                is_noop = True
                            elif val[0] == "K" and cval[0] == "K" and val[1] == cval[1] and val[2] == cval[2]:
                                is_noop = True
                        if is_noop:
                            st["st_home_noop" if sp == "HOME" else "st_l1_noop"] += 1
                            continue

                        snid = add_node("st_home" if sp == "HOME" else "st_l1", reg, ph, [pn], 1, loc)
                        pass_store_nodes.append(snid)
                        mem_set(loc, val, snid)
                pass_records.append((reg, ph, pass_store_nodes))

    # Close out active solve if any
    if active_solve is not None and active_solve.startswith("solve"):
        for loc in written_by_reg[active_solve]:
            if loc in mem and mem[loc][2] is not None:
                roots.append(mem[loc][2])

    # Mark live-out outer state locations as roots
    for loc in state_locs:
        if loc in mem and mem[loc][2] is not None:
            roots.append(mem[loc][2])
            if mem[loc][0][0] == "K" or (mem[loc][0][0] == "V" and len(mem[loc][0][2]) > 1):
                block_reasons["live_out_state"] += 1

    # Also mark any final value in mem that is read-before-write in outer (loop-carried across outer iterations!)
    # Check which locations are read at version 0 in mem_get (i.e. read before first write in outer):
    # Actually: any location in `written_all` that was read before written in outer IS loop-carried across outer!
    # Our state_locs covers all prog.state registers, which are the ONLY outer loop-carried variables in SimpleFoam.
    # Let's assert/verify by checking if any non-state loc in written_all was read before written in outer!

    # Backward liveness reachability from roots
    live = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid in live:
            continue
        live.add(nid)
        for d in nodes[nid][3]:
            if d not in live:
                stack.append(d)

    # Tally live nodes by (reg, ph)
    live_stats = defaultdict(Counter)
    live_op_by_name = defaultdict(Counter)
    for nid, (cat, reg, ph, deps, weight, meta) in enumerate(nodes):
        if nid not in live:
            continue
        ls = live_stats[(reg, ph)]
        if cat == "op":
            ls["op_tiles_live"] += 1
            live_op_by_name[(reg, ph)][meta] += 1
        elif cat == "ldr_const":
            ls["const_pages_live"] += 1
            ls["stream_pages_live"] += 1
        elif cat == "ldr_const_pat":
            ls["const_pages_live"] += 1
            ls["stream_pages_live"] += 1
            ls["patched_vals_live"] += meta
            ls["gathered_vals_live"] += meta
        elif cat == "ldr_home":
            ls["ldr_home_live"] += 1
            ls["stream_pages_live"] += 1
        elif cat == "ldr_home_pat":
            ls["ldr_home_live"] += 1
            ls["stream_pages_live"] += 1
            ls["patched_vals_live"] += meta
            ls["gathered_vals_live"] += meta
        elif cat == "ldr_zero":
            ls["ldr_zero_live"] += 1
            ls["stream_pages_live"] += 1
            ls["patched_vals_live"] += meta
            ls["gathered_vals_live"] += meta
        elif cat == "gather_tile":
            ls["gather_tiles_live"] += 1
            ls["gather_vals_live"] += weight
            ls["gathered_vals_live"] += weight
        elif cat == "st_home":
            ls["st_home_live"] += 1
        elif cat == "st_l1":
            ls["st_l1_live"] += 1
        elif cat == "ld_l1":
            ls["ld_l1_live"] += 1
        elif cat == "lds":
            ls["lds_l1_live"] += 1
        elif cat == "cpd":
            ls["cpds_live"] += 1
        elif cat in ("fetch_const", "fetch_const_uni"):
            ls["fetch_pages_live"] += 1

    # Count completely dead tile-passes (no live store in the pass)
    for reg, ph, snids in pass_records:
        if not any(s in live for s in snids):
            stats[(reg, ph)]["passes_dead"] += 1

    # Build clean structured summary
    rows = {}
    for key in sorted(set(stats) | set(live_stats)):
        st = stats[key]
        ls = live_stats[key]
        op_tot = st["op_tiles"]
        op_live = ls["op_tiles_live"]
        g_tot = st["gathered_vals"]
        g_live = ls["gathered_vals_live"]
        sp_tot = st["stream_pages"]
        sp_live = ls["stream_pages_live"]
        cp_tot = st["const_pages"]
        cp_live = ls["const_pages_live"]
        cp_uni = st["const_pages_uniform"]
        # Non-const stream pages (home + l1 + zero/gather)
        nc_tot = sp_tot - cp_tot
        nc_live = sp_live - cp_live
        # Incremental const pages killed beyond --kfold (non-uniform const pages now dead)
        cp_nonuni_dead = (cp_tot - cp_uni) - cp_live
        # Per-op-name dead counts
        op_dead_by_name = {}
        for k, v in st.items():
            if k.startswith("op:"):
                nm = k[3:]
                op_dead_by_name[nm] = (v - live_op_by_name[key][nm], v)
        rows[f"{key[0]}:{key[1]}"] = dict(
            region=key[0],
            phase=key[1],
            passes=int(st["passes"]),
            passes_dead=int(st["passes_dead"]),
            op_tiles=int(op_tot),
            op_tiles_dead=int(op_tot - op_live),
            op_by_name=op_dead_by_name,
            gathered_vals=int(g_tot),
            gathered_vals_dead=int(g_tot - g_live),
            patched_vals=int(st["patched_vals"]),
            patched_vals_dead=int(st["patched_vals"] - ls["patched_vals_live"]),
            gather_vals=int(st["gather_vals"]),
            gather_vals_dead=int(st["gather_vals"] - ls["gather_vals_live"]),
            stream_pages=int(sp_tot),
            stream_pages_dead=int(sp_tot - sp_live),
            nonconst_stream_pages=int(nc_tot),
            nonconst_stream_pages_dead=int(nc_tot - nc_live),
            const_pages=int(cp_tot),
            const_pages_dead=int(cp_tot - cp_live),
            const_pages_uniform=int(cp_uni),
            const_pages_nonuni_dead=int(cp_nonuni_dead),
            st_home=int(st["st_home"]),
            st_home_dead=int(st["st_home"] - ls["st_home_live"]),
            st_home_noop=int(st["st_home_noop"]),
            st_l1=int(st["st_l1"]),
            st_l1_dead=int(st["st_l1"] - ls["st_l1_live"]),
            ld_l1=int(st["ld_l1"]),
            ld_l1_dead=int(st["ld_l1"] - ls["ld_l1_live"]),
            fetch_pages=int(st["fetch_pages"]),
            fetch_pages_dead=int(st["fetch_pages"] - ls["fetch_pages_live"]),
            op_dead_const=int(st["op_dead_const"]),
            op_dead_ident_exact=int(st["op_dead_ident_exact"]),
            op_dead_ident_care=int(st["op_dead_ident_care"]),
            op_dead_zero_care=int(st["op_dead_zero_care"]),
        )

    return dict(
        name=prog["name"],
        core=prog["core"],
        T=T,
        cells=prog["cells"],
        care=care,
        use_ro_nonconst=use_ro_nonconst,
        rows=rows,
        block_reasons=dict(block_reasons),
    )


def format_report(res):
    lines = []
    lines.append(f"=== {res['name']} core={res['core']} T={res['T']} cells={res['cells']} care={res['care']} ro_nonconst={res['use_ro_nonconst']} ===")
    hdr = f"{'region:phase':22s} | {'op-tile dead/tot (%)':24s} | {'gather dead/tot (%)':24s} | {'stream dead/tot (%)':24s} | {'const dead/tot (%)':24s} | {'st_home dead/tot':18s} | {'passes dead/tot':16s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def fmt(d, t):
        pct = 100.0 * d / t if t else 0.0
        return f"{d:7d}/{t:7d} ({pct:5.1f}%)"

    outer_tot = Counter()
    for rk, r in res["rows"].items():
        if r["region"] == "outer":
            for k in ("op_tiles", "op_tiles_dead", "gathered_vals", "gathered_vals_dead",
                      "stream_pages", "stream_pages_dead", "nonconst_stream_pages", "nonconst_stream_pages_dead",
                      "const_pages", "const_pages_dead", "const_pages_uniform", "const_pages_nonuni_dead",
                      "st_home", "st_home_dead", "passes", "passes_dead"):
                outer_tot[k] += r[k]
        lines.append(
            f"{rk:22s} | {fmt(r['op_tiles_dead'], r['op_tiles']):24s} | "
            f"{fmt(r['gathered_vals_dead'], r['gathered_vals']):24s} | "
            f"{fmt(r['stream_pages_dead'], r['stream_pages']):24s} | "
            f"{fmt(r['const_pages_dead'], r['const_pages']):24s} | "
            f"{r['st_home_dead']:6d}/{r['st_home']:6d}     | "
            f"{r['passes_dead']:6d}/{r['passes']:6d}"
        )
    lines.append("-" * len(hdr))
    lines.append(
        f"{'OUTER TOTAL':22s} | {fmt(outer_tot['op_tiles_dead'], outer_tot['op_tiles']):24s} | "
        f"{fmt(outer_tot['gathered_vals_dead'], outer_tot['gathered_vals']):24s} | "
        f"{fmt(outer_tot['stream_pages_dead'], outer_tot['stream_pages']):24s} | "
        f"{fmt(outer_tot['const_pages_dead'], outer_tot['const_pages']):24s} | "
        f"{outer_tot['st_home_dead']:6d}/{outer_tot['st_home']:6d}     | "
        f"{outer_tot['passes_dead']:6d}/{outer_tot['passes']:6d}"
    )
    lines.append(
        f"  OUTER non-const stream dead: {fmt(outer_tot['nonconst_stream_pages_dead'], outer_tot['nonconst_stream_pages'])}; "
        f"const uniform (kfold): {outer_tot['const_pages_uniform']}/{outer_tot['const_pages']} "
        f"({100*outer_tot['const_pages_uniform']/max(outer_tot['const_pages'],1):.1f}%); "
        f"extra non-uniform const dead: {outer_tot['const_pages_nonuni_dead']}"
    )
    lines.append(f"  Store elimination blocks: {res['block_reasons']}")
    return "\n".join(lines)


# ==================================================================================================
# Self-tests (including broken-case / blocking-case checks)
# ==================================================================================================

def run_self_tests():
    # 2 tiles: tile 0 has const0 = 0.0, const1 = 1.0, const2 = non-uniform; tile 1 has all non-uniform.
    const_uni = np.array([ZERO_POS, -1, ONE_U32, -1, -1, -1], np.int64)  # c0=[0..1], c1=[2..3], c2=[4..5]
    # Test 1: Chain x*0 -> zero -> y+zero -> copy(y) -> store -> consumer takes y directly
    #         Also test patched ldr dead when multiplied by zero-page.
    map_counts = {0: (np.array([100, 200]), np.array([10, 20]), np.array([5, 5]), False)}
    events = [
        # blk 0: t0 = ldr(HOME 0, patched map 0) * ldr(CONST 0) -> st(HOME 10)
        #        tile 0: const0[0]=0 -> mul dead, patched ldr dead (115 vals), HOME 10[0]=0!
        #        tile 1: const0[1]=non-uni -> all live!
        (0, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 0), 0, 0, [(("HOME", 0), None)], False, False),
             ("ldr", 1, ("CONST", 0), 0, None, [], False, False),
             ("op", "mul", 2, 0, 1)],
            [(2, ("HOME", 10, True))]
        )])),
        # blk 1: t1 = ldr(HOME 2) + ldr(HOME 10) -> st(HOME 12)
        #        tile 0: HOME 10[0]=0 -> add dead, ldr(HOME 10) dead, HOME 12[0] = copy(HOME 2[0])!
        (1, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 2), 0, None, [], False, False),
             ("ldr", 1, ("HOME", 10), 0, None, [], False, False),
             ("op", "add", 2, 0, 1)],
            [(2, ("HOME", 12, True))]
        )])),
        # blk 2: out_state = ldr(HOME 12) * ldr(CONST 2 [=1.0 on tile 0]) -> st(HOME 20 [live-out])
        #        tile 0: const1[0]=1.0 -> mul dead! HOME 20 is live-out, so st(HOME 20) stays LIVE,
        #        and reads HOME 2[0] directly (bypassing HOME 12[0]), so st(HOME 12)[0] AND blk 1's ldr(HOME 2)[0] are DEAD!
        (2, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 12), 0, None, [], False, False),
             ("ldr", 1, ("CONST", 2), 0, None, [], False, False),
             ("op", "mul", 2, 0, 1)],
            [(2, ("HOME", 20, True))]
        )])),
    ]
    prog = dict(
        name="selftest1", core=0, T=2, cells=2048, cpages=6,
        const_uni=const_uni, init_ro={}, state_locs={("HOME", 20), ("HOME", 21)},
        map_counts=map_counts, events=events, flags={},
    )
    r_all = analyze_region(prog, care=True)
    row = r_all["rows"]["outer:mom"]
    assert row["op_tiles"] == 6 and row["op_tiles_dead"] == 3, row
    assert row["gathered_vals"] == 340 and row["gathered_vals_dead"] == 115, row
    # Tile 0 live stream pages: only blk 2's ldr(HOME 2)[0] = 1 live (out of 6 on tile 0 -> 5 dead on tile 0!)
    assert row["stream_pages"] == 12 and row["stream_pages_dead"] == 5, row
    assert row["st_home"] == 6 and row["st_home_dead"] == 2, row  # HOME 10[0] and HOME 12[0] dead; HOME 20[0] live-out!
    assert row["passes"] == 6 and row["passes_dead"] == 2, row    # blk 0 tile 0 and blk 1 tile 0 completely dead!

    # Exact mode (care=False): x*0 and y+(+0) must NOT fold, ONLY x*1 folds!
    r_ex = analyze_region(prog, care=False)
    row_ex = r_ex["rows"]["outer:mom"]
    assert row_ex["op_tiles_dead"] == 1, row_ex  # only mul by 1.0 on tile 0!
    assert row_ex["gathered_vals_dead"] == 0, row_ex

    # Test 2: Source clobbered before reader -> copy store MUST stay LIVE and log source_clobbered!
    events2 = [
        # blk 0: HOME 12 = HOME 2 * 1.0 (tile 0 is copy of HOME 2)
        (0, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 2), 0, None, [], False, False),
             ("ldr", 1, ("CONST", 2), 0, None, [], False, False),
             ("op", "mul", 2, 0, 1)],
            [(2, ("HOME", 12, True))]
        )])),
        # blk 1: CLOBBER HOME 2 with new value (ldr HOME 4 + ldr HOME 4)
        (1, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 4), 0, None, [], False, False),
             ("op", "add", 1, 0, 0)],
            [(1, ("HOME", 2, True))]
        )])),
        # blk 2: live-out HOME 20 = ldr(HOME 12) + ldr(HOME 4) -> CANNOT bypass HOME 12 because HOME 2 was clobbered!
        (2, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 12), 0, None, [], False, False),
             ("ldr", 1, ("HOME", 4), 0, None, [], False, False),
             ("op", "add", 2, 0, 1)],
            [(2, ("HOME", 20, True))]
        )])),
    ]
    prog2 = dict(
        name="selftest2_clobber", core=0, T=2, cells=2048, cpages=6,
        const_uni=const_uni, init_ro={}, state_locs={("HOME", 20), ("HOME", 21)},
        map_counts={}, events=events2, flags={},
    )
    r2 = analyze_region(prog2, care=True)
    row2 = r2["rows"]["outer:mom"]
    # mul*1 is dead (1 dead op), blk 1 clobbers HOME 2, so HOME 12 store MUST stay live!
    # Also blk 1's store to HOME 2 is itself dead (no reader after clobber!) -> wait: HOME 2 is clobbered
    # at blk 1, so at blk 2 mem[HOME 2].ver != v0 -> source_clobbered!
    assert row2["st_home_dead"] == 2, row2  # blk 1's 2 dead stores to clobbered HOME 2; blk 0's HOME 12[0] stays LIVE!
    assert r2["block_reasons"].get("source_clobbered", 0) >= 1, r2["block_reasons"]

    # Test 3: Broken check — if const0[0] is non-uniform instead of zero, dead ops drop from 3 to 1
    const_uni_broken = const_uni.copy()
    const_uni_broken[0] = -1
    prog3 = dict(prog, const_uni=const_uni_broken)
    r3 = analyze_region(prog3, care=True)
    assert r3["rows"]["outer:mom"]["op_tiles_dead"] == 1, r3["rows"]["outer:mom"]

    # Test 4: Split gather (lds -> L1 5, then partial non-dense gather -> L1 5, then mul by const0 -> HOME 20)
    # On tile 1 (const0[1] non-uniform, live), gather preserves untouched lanes so lds MUST stay LIVE!
    # On tile 0 (const0[0] = 0, dead), BOTH gather (115 vals) AND lds pass MUST become DEAD!
    events4 = [
        (0, "outer", "mom", ("blk_run", 2, [(
            [("lds", 0, ("L1", 10, True), 4)],
            [(0, ("L1", 5, True))]
        )])),
        (1, "outer", "mom", ("gather", 5, 0, [(("L1", 10), None)], 0)),
        (2, "outer", "mom", ("blk_run", 2, [(
            [("ld", 0, ("L1", 5, True)),
             ("ldr", 1, ("CONST", 0), 0, None, [], False, False),
             ("op", "mul", 2, 0, 1)],
            [(2, ("HOME", 20, True))]
        )])),
    ]
    prog4 = dict(
        name="selftest4_split_gather", core=0, T=2, cells=2048, cpages=6,
        const_uni=const_uni, init_ro={}, state_locs={("HOME", 20), ("HOME", 21)},
        map_counts=map_counts, events=events4, flags={},
    )
    r4 = analyze_region(prog4, care=True)
    row4 = r4["rows"]["outer:mom"]
    assert row4["passes"] == 4 and row4["passes_dead"] == 1, row4  # only lds tile 0 dead; lds tile 1 LIVE!
    assert row4["gathered_vals"] == 340 and row4["gathered_vals_dead"] == 115, row4
    r4_ex = analyze_region(prog4, care=False)
    assert r4_ex["rows"]["outer:mom"]["passes_dead"] == 0, r4_ex["rows"]["outer:mom"]
    print("SELF-TEST PASS (all 4 cases + exact mode + clobber guard + split-gather + broken check)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="const_census extract (*.cx.zst) or saved Image directory")
    ap.add_argument("--aux", help="optional aux.npz sidecar for *.cx.zst")
    ap.add_argument("--core", default="0")
    ap.add_argument("--exact", action="store_true", help="only bitwise-exact identities (care=False)")
    ap.add_argument("--both", action="store_true", help="run both exact (care=False) and all identities (care=True)")
    ap.add_argument("--no-ro-nonconst", action="store_true", help="ignore read-only pre-initialized L1/HOME tiles (like cell_mask)")
    ap.add_argument("--json", help="write full JSON output")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        run_self_tests()
        if not a.src:
            return

    cores = [int(c) for c in a.core.split(",")]
    modes = [False, True] if a.both else [not a.exact]
    results = []
    for c in cores:
        if a.src.endswith(".cx.zst"):
            prog = load_from_extract(a.src, c, aux_path=a.aux)
        else:
            prog = load_from_image(a.src, c)
        for care in modes:
            res = analyze_region(prog, care=care, use_ro_nonconst=not a.no_ro_nonconst)
            print(format_report(res), flush=True)
            results.append(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
