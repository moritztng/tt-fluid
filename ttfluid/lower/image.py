"""Per-chip device images of a lowered mesh program: what run_mesh.py uploads to each chip, one file per chip, so a large
case is never held as one Machine + Encoded (an occDrivAer 65M Encoded is ~200 GB) and the device run reads only its
chips' bytes.

    img = Image.from_lowering(m, E, tab, gtab, info, K, P, flags)   # from a global lowering (run_mesh.py)
    img.write(d, floor_gb)  # refuses when the write would leave the disk under floor_gb free; d/shared.npz + d/chip<k>/<array>.npy (raw .npy: the loader memory-maps them, so the
                            # host holds file-backed pages the kernel can evict, not K chips of anonymous memory)
    img = Image.load(d)     # the same object back, chips read lazily
    img.chip(k)             # dict of chip k's arrays, exactly as the device gets them (tab before the gateway words)
    img.digest(k)           # sha256 over chip k's arrays and the shared ones: the exactness gate between lowerings

Shared: the three code streams, the scalars the kernels are built with, the gateway tables (global core numbers), the
output names and the state layout. Per chip k (cores k*P .. k*P+P-1): arena rows, DRAM homes, streamed constants,
residue/halo tables (padded to the global width L0), table blob pages, the halo table offsets, and for every state
space the global ids of the elements it owns with their (core, lane), so results are collected chip by chip.
"""
import functools
import hashlib
import io
import json
import mmap
import os
import shutil
import struct
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LANES = 1024
PAGE = 4096
SCALARS = ("K", "P", "T", "ndram", "hpages", "cpages", "ntiles", "nhalo", "hmax", "red2", "redm", "L0", "TBP", "tab_stream", "nmaxop",
           "loop_pages")


def par_zero(arr):
    """Zero a C-contiguous array across _READERS when large enough to pay for the fan-out."""
    if arr.nbytes < (1 << 22):
        arr.fill(0)
        return
    flat = arr.reshape(-1)
    nw = max(1, _READERS._max_workers)
    step = -(-len(flat) // nw)
    list(_READERS.map(lambda i: flat[i:i + step].fill(0), range(0, len(flat), step)))


def load_state(path):
    """Read a state .npz as name -> array (dropping hist). Uncompressed .npz files (as np.savez writes) are
    memory-mapped directly without copying into anonymous RAM; compressed archives fall back to np.load."""
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zf:
            infos = [info for info in zf.infolist() if info.filename != "hist.npy"]
            if all(info.compress_type == zipfile.ZIP_STORED and info.filename.endswith(".npy") for info in infos):
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                out = {}
                ok = True
                for info in infos:
                    name = info.filename[:-4]
                    fname_len, extra_len = struct.unpack_from("<HH", mm, info.header_offset + 26)
                    f.seek(info.header_offset + 30 + fname_len + extra_len)
                    major, _ = np.lib.format.read_magic(f)
                    if major not in (1, 2):
                        ok = False
                        break
                    hdr = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                    shape, fortran_order, dtype = hdr(f)
                    out[name] = np.ndarray(shape, dtype=dtype, buffer=mm, offset=f.tell(),
                                           order="F" if fortran_order else "C")
                if ok:
                    return out
                mm.close()
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "hist"}


class Pages:
    """A (cores, pages, LANES) array held as the pages that are not zero: rows[row[core, page]] is a page, and the
    pages left out share one row on the way back from a chip (zero) and have row -1 on the way in, where a write to
    them is dropped (they are zero, or another part of the upload holds them). Indexed as Image.state and put_state index the homes: [core, page, lane] with index arrays."""

    def __init__(self, rows, row, zero_row=None):
        self.rows, self.row, self.zero_row = rows, row, zero_row

    def view(self, dtype):
        return Pages(self.rows.view(dtype), self.row, self.zero_row)

    def __getitem__(self, i):
        return self.rows[self.row[i[0], i[1]], i[2]]

    def __setitem__(self, i, v):
        r = self.row[i[0], i[1]]
        m = r >= 0           # row -1: the page is held elsewhere (another part of a staged upload), not written here
        if m.all():
            self.rows[r, i[2]] = v
        else:
            self.rows[r[m], i[2][m]] = v[m]


def _write_core(X, c, tt, ntc, nc, col):
    if isinstance(X, Pages):
        rr = X.row[c, tt:tt + ntc]
        if (rr >= 0).all() and (ntc <= 1 or (rr[1:] == rr[:-1] + 1).all()):
            X.rows[int(rr[0]):int(rr[0]) + ntc].ravel()[:nc] = col
        else:
            for j in range(ntc):
                rj = int(rr[j])
                if rj >= 0:
                    lo = j * LANES
                    hi = min(lo + LANES, nc)
                    X.rows[rj, :hi - lo] = col[lo:hi]
    else:
        X[c, tt:tt + ntc].ravel()[:nc] = col


def _read_core(X, c, tt, ntc, nc, dst):
    if isinstance(X, Pages):
        rr = X.row[c, tt:tt + ntc]
        zr = X.zero_row
        if (zr is None or (rr != zr).all()) and (ntc <= 1 or (rr[1:] == rr[:-1] + 1).all()):
            dst[:] = X.rows[int(rr[0]):int(rr[0]) + ntc].ravel()[:nc]
        else:
            for j in range(ntc):
                rj = int(rr[j])
                if zr is None or rj != zr:
                    lo = j * LANES
                    hi = min(lo + LANES, nc)
                    dst[lo:hi] = X.rows[rj, :hi - lo]
    else:
        dst[:] = X[c, tt:tt + ntc].ravel()[:nc]


class Image:
    def __init__(self, shared, chips):
        self.shared = shared          # name -> array or JSON-able object
        self._chips = chips           # k -> dict, or k -> path (lazy)
        self._members = {}            # (k, space) -> members(k, space)
        self._layouts = {}            # (k, space) -> compact per-core layout
        self._mlock = threading.Lock()
        self.s = shared["scalars"]

    # ---- from a global lowering --------------------------------------------------------------------------------
    @classmethod
    def from_lowering(cls, m, E, tab, gtab, info, K, P, flags):
        L0 = max(len(t) for t in tab)
        shared = cls.shared_of(m, E, gtab, info, K, P, flags, L0)
        own = {name: np.asarray(m.lay.owner[name]) for name in shared["spaces"]}

        def members(name, k):
            ids = np.nonzero(own[name] // P == k)[0]
            return ids, own[name][ids] - k * P, np.asarray(m.lay.lane[name][ids])
        chips = {k: cls.chip_of(E, tab, slice(k * P, (k + 1) * P), L0, shared, lambda name: members(name, k))
                 for k in range(K)}
        return cls(shared, chips)

    @staticmethod
    def shared_of(m, E, gtab, info, K, P, flags, L0, space_n=None, const_state=None):
        """Everything every chip shares. space_n: global element count per space (default the Machine's own, a
        global lowering); const_state: name/i -> the global array of a state register that stayed a constant
        (default from the Machine's prog_pre)."""
        TBP = -(-E.tab_blob.shape[1] // LANES) if E.tab_stream else 1
        sc = dict(K=K, P=P, T=int(m.T), ndram=int(m.ndram), hpages=int(m.hpages), cpages=int(m.cpages), ntiles=int(m.ntiles), nhalo=int(m.nhalo),
                  hmax=int(m.maps.hmax), red2=bool(m.red2), redm=bool(m.redm), L0=L0, TBP=TBP,
                  tab_stream=bool(E.tab_stream), nmaxop=int(E.n_maxop()), loop_pages=int(E.loop_pages))
        if len(getattr(m.maps, "plans", ())) > 1:      # multigrid halo plans: slot sizes per plan (Machine.hoff)
            sc.update(nplans=len(m.maps.plans), hwords=int(m.hwords))
        # the state: per name the registers' (phys, tiles, space) or a constant (prog_pre) value
        state, spaces = [], {}
        for name, cr in m.prog.state:
            regs = []
            for r in cr.ravel():
                r2 = m.ren(r)
                sp = m.space[r2]
                spaces[sp.name] = int(sp.n if space_n is None else space_n[sp.name])
                regs.append([sp.name, int(m.phys[r2]), int(m.size(r2))] if r2 in m.phys else [sp.name, None, None])
            state.append([name, list(cr.shape), regs])
        if const_state is None:
            const_state = {f"{name}/{i}": np.asarray(m.prog_pre[m.ren(r)], np.float32)
                           for name, cr in m.prog.state for i, r in enumerate(cr.ravel()) if m.ren(r) not in m.phys}
        sizes = dict(E.sizes(), tab_words=L0)
        return dict(scalars=sc, flags=flags, info=info, sizes=sizes, out_names=list(E.out_names),
                    gtab={str(g): v for g, v in gtab.items()}, state=state, spaces=spaces,
                    c=np.asarray(E.c, np.uint32), b=np.asarray(E.b, np.uint32), n=np.asarray(E.n, np.uint32),
                    const_state=const_state)

    @staticmethod
    def chip_of(E, tab, rows, L0, shared, members, arrays=None):
        """One chip's arrays: E's rows `rows` (a slice over E's cores), tab its rewritten table rows (lists, indexed
        like E's), members(space) -> (global ids, core on the chip, lane) of the elements the chip owns.
        `arrays`, when given, supplies (arena, home, consts) directly instead of slicing E."""
        TBP = shared["scalars"]["TBP"]
        t = np.zeros((rows.stop - rows.start, L0), np.uint32)
        for i, row in enumerate(tab[rows]):
            t[i, :len(row)] = row
        A, Hm, C = (E.arena[rows], E.home[rows], E.consts[rows]) if arrays is None else arrays
        ch = dict(arena=np.ascontiguousarray(A), home=np.ascontiguousarray(Hm),
                  consts=np.ascontiguousarray(C), tab=t,
                  tab_halo=np.asarray(E.tab_halo[rows], np.uint32))
        if E.tab_stream:
            blob = np.zeros((rows.stop - rows.start, TBP * LANES), np.uint32)
            blob[:, :E.tab_blob.shape[1]] = E.tab_blob[rows]
            ch["blob"] = blob
        for name in shared["spaces"]:
            ids, core, lane = members(name)
            ch["ids/" + name] = np.asarray(ids, np.int64)
            ch["core/" + name] = np.asarray(core, np.int32)
            ch["lane/" + name] = np.asarray(lane, np.int32)
        return ch

    @staticmethod
    def used_chip(ch):
        """Non-zero 4-KB blocks of each array in one chip's dict, scanned in parallel across arrays."""
        items = list(ch.items())
        return dict(zip((n for n, _ in items), _READERS.map(lambda kv: _used_pages(kv[1]), items)))

    @staticmethod
    def write_chip_arrays(d, k, ch, used_k=None):
        """Write one chip's arrays as sparse .npy files in parallel across arrays."""
        cd = Path(d) / f"chip{k}"
        cd.mkdir(parents=True, exist_ok=True)

        def _one(kv):
            name, arr = kv
            f = cd / (name.replace("/", "@") + ".npy")
            f.unlink(missing_ok=True)   # never truncate in place: identical arrays are hardlinked across images
            _save_sparse(f, arr, used_k and used_k[name])

        list(_READERS.map(_one, list(ch.items())))

    @staticmethod
    def digest_chip(shared, ch):
        h = hashlib.sha256()
        for name in ("c", "b", "n"):
            h.update(name.encode() + np.ascontiguousarray(shared[name]).tobytes())
        h.update(json.dumps([shared["scalars"], shared["gtab"], shared["out_names"]], sort_keys=True).encode())
        for name in sorted(ch):
            a = np.ascontiguousarray(ch[name])
            h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.reshape(-1).view(np.uint8))   # the array's bytes without the copy tobytes() makes (home is 8 GB a chip on Small)
        return h.hexdigest()

    def write_shared(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.shared.items() if isinstance(v, np.ndarray)}
        arrays.update({"const_state/" + k: v for k, v in self.shared["const_state"].items()})
        meta = {k: v for k, v in self.shared.items() if not isinstance(v, np.ndarray) and k != "const_state"}
        np.savez(d / "shared.npz", meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)

    def write_chip(self, d, k, used=None):
        (Path(d) / f"chip{k}").mkdir(parents=True, exist_ok=True)
        for name, arr in self.chip(k).items():
            f = Path(d) / f"chip{k}" / (name.replace("/", "@") + ".npy")
            f.unlink(missing_ok=True)   # never truncate in place: identical arrays are hardlinked across images
            _save_sparse(f, arr, used and used[k, name])

    # ---- files ---------------------------------------------------------------------------------------------------
    def used(self):
        """Every chip array's non-zero 4-KB blocks, found once: the disk check counts them and the write stores only those."""
        K = self.s["K"]
        with ThreadPoolExecutor(min(K, 8)) as ex:
            return dict(kv for kvs in ex.map(lambda k: [((k, n), _used_pages(v)) for n, v in self.chip(k).items()], range(K)) for kv in kvs)

    @staticmethod
    def disk_bytes(used):
        return sum(1 + int(u.sum()) for u in used.values()) * PAGE

    def write(self, d, floor_gb=0, used=None):
        """The files are np.save's byte for byte, with the zero pages as holes (home 97 %, consts 58 % on Small k4)."""
        used = self.used() if used is None else used
        check_free(d, self.disk_bytes(used), floor_gb)
        self.write_shared(d)
        with ThreadPoolExecutor(min(self.s["K"], 8)) as ex:
            list(ex.map(lambda k: self.write_chip(d, k, used), range(self.s["K"])))

    @classmethod
    def load(cls, d):
        d = Path(d)
        z = np.load(d / "shared.npz")
        shared = json.loads(bytes(z["meta"]).decode())
        for k in z.files:
            if k.startswith("const_state/"):
                shared.setdefault("const_state", {})[k[len("const_state/"):]] = z[k]
            elif k != "meta":
                shared[k] = z[k]
        shared.setdefault("const_state", {})
        return cls(shared, {k: d / f"chip{k}" for k in range(shared["scalars"]["K"])})

    def chip(self, k):
        c = self._chips[k]
        return _ChipFiles(c) if isinstance(c, Path) else c

    # ---- the gate ------------------------------------------------------------------------------------------------
    def digest(self, k):
        return Image.digest_chip(self.shared, self.chip(k))

    # ---- results, chip by chip ------------------------------------------------------------------------------------
    def _layout(self, k, spn):
        """Compact per-core layout for chip k's elements of space spn. When every core's lanes are 0..n_c-1
        (the normal lowering output), returns (True, ptr[P+1], ids_sorted) in O(N) without sorting or storing
        core/tile/lane arrays; otherwise falls back to (False, core, tile, lane, ids) in (core, lane) order."""
        key = (k, spn)
        with self._mlock:
            lay = self._layouts.get(key)
        if lay is not None:
            return lay
        ch = self.chip(k)
        core = np.asarray(ch[f"core/{spn}"])
        lane = np.asarray(ch[f"lane/{spn}"])
        ids = np.asarray(ch[f"ids/{spn}"])
        P = int(self.s["P"])
        n = len(ids)
        lay = None
        if n == 0:
            lay = (True, np.zeros(P + 1, np.int64), np.empty(0, np.uint32))
        elif int(core.min()) >= 0 and int(core.max()) < P:
            cnt = np.bincount(core, minlength=P)
            ptr = np.empty(P + 1, np.int64)
            ptr[0] = 0
            np.cumsum(cnt, out=ptr[1:])
            if int(lane.min()) >= 0 and (lane < cnt[core]).all():
                dest = ptr[core] + lane
                seen = np.zeros(n, np.uint8)
                seen[dest] = 1
                if seen.all():
                    id_dt = np.uint32 if int(self.shared["spaces"][spn]) < (1 << 32) else np.int64
                    ids_s = np.empty(n, id_dt)
                    ids_s[dest] = ids
                    lay = (True, ptr, ids_s)
        if lay is None:
            o = np.argsort(core.astype(np.int64) << 32 | lane, kind="stable")
            lay = (False, core[o].astype(np.int32), (lane[o] // LANES).astype(np.int32),
                   (lane[o] % LANES).astype(np.int32), ids[o])
        with self._mlock:
            return self._layouts.setdefault(key, lay)

    def _warm_layouts(self, keys):
        with self._mlock:
            todo = [key for key in keys if key not in self._layouts]
        if len(todo) > 1:
            list(_READERS.map(lambda kv: self._layout(*kv), todo))
        elif todo:
            self._layout(*todo[0])

    def state(self, out_rows, home_rows):
        """out_rows(k) -> chip k's arena (P, ntiles, LANES) fp32; home_rows(k) -> its DRAM homes (P, hpages, LANES) or Pages.
        Returns name -> global array, as Encoded.state_from_arena. Contiguous per-core layouts scatter in core chunks
        across _READERS and skip pages marked zero."""
        from ttfluid.resident.machine import DBASE
        K, P = int(self.s["K"]), int(self.s["P"])
        out = {}
        spaces = set()
        for name, shape, regs in self.shared["state"]:
            out[name] = np.zeros((self.shared["spaces"][regs[0][0]], len(regs)), np.float32)
            for i, (spn, t, n) in enumerate(regs):
                if t is None:
                    out[name][:, i] = self.shared["const_state"][f"{name}/{i}"]
                else:
                    spaces.add(spn)
        self._warm_layouts([(k, spn) for k in range(K) for spn in spaces])

        def get_general(col, ids, X, idx):
            col[ids] = X[idx]

        def get_chunk(out_arr, ncols, all_dev, reg_ops, ptr, ids_s, c0, c1):
            for c in range(c0, c1):
                a0, a1 = int(ptr[c]), int(ptr[c + 1])
                if a0 == a1:
                    continue
                nc = a1 - a0
                ntc = -(-nc // LANES)
                active = []
                for i, X, tt in reg_ops:
                    if isinstance(X, Pages) and X.zero_row is not None:
                        if (X.row[c, tt:tt + ntc] == X.zero_row).all():
                            continue
                    active.append((i, X, tt))
                if not active:
                    continue
                idx_ids = ids_s[a0:a1]
                if ncols == 1:
                    _, X, tt = active[0]
                    col = np.zeros(nc, np.float32)
                    _read_core(X, c, tt, ntc, nc, col)
                    out_arr[idx_ids, 0] = col
                elif all_dev:
                    buf = np.zeros((nc, ncols), np.float32)
                    for i, X, tt in active:
                        _read_core(X, c, tt, ntc, nc, buf[:, i])
                    out_arr[idx_ids] = buf
                else:
                    col = np.zeros(nc, np.float32)
                    for i, X, tt in active:
                        col.fill(0)
                        _read_core(X, c, tt, ntc, nc, col)
                        out_arr[idx_ids, i] = col

        ah = [(out_rows(k), home_rows(k)) for k in range(K)]
        step = max(1, -(-P // max(1, _READERS._max_workers)))
        jobs = []
        for k in range(K):
            A, H = ah[k]
            for name, shape, regs in self.shared["state"]:
                reg_ops = [(i, H if t >= DBASE else A, t - DBASE if t >= DBASE else t)
                           for i, (spn, t, n) in enumerate(regs) if t is not None]
                if not reg_ops:
                    continue
                spn0 = regs[reg_ops[0][0]][0]
                same_sp = all(regs[i][0] == spn0 for i, _, _ in reg_ops)
                lay = self._layout(k, spn0) if same_sp else None
                if lay is not None and lay[0]:
                    _, ptr, ids_s = lay
                    all_dev = len(reg_ops) == len(regs)
                    for c0 in range(0, P, step):
                        jobs.append(_READERS.submit(get_chunk, out[name], len(regs), all_dev, reg_ops,
                                                    ptr, ids_s, c0, min(c0 + step, P)))
                else:
                    for i, X, tt in reg_ops:
                        core, tile, lane, ids = self.members(k, regs[i][0])
                        jobs.append(_READERS.submit(get_general, out[name][:, i], ids, X, (core, tt + tile, lane)))
        for j in jobs:
            j.result()
        return {name: out[name].reshape((-1,) + tuple(shape)) if shape else out[name][:, 0]
                for name, shape, regs in self.shared["state"]}

    def members(self, k, spn):
        """Chip k's elements of space spn in the order they lie on the chip: (core, tile, lane in the tile, global id),
        so a register's values are written front to back (in global-id order every write lands on another page)."""
        if (k, spn) not in self._members:
            lay = self._layout(k, spn)
            if lay[0]:
                _, ptr, ids_s = lay
                counts = np.diff(ptr)
                core = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
                lane = np.arange(len(ids_s), dtype=np.int32) - np.repeat(ptr[:-1], counts).astype(np.int32)
                self._members[k, spn] = core, lane // LANES, lane % LANES, ids_s.astype(np.int64)
            else:
                self._members[k, spn] = lay[1:]
        return self._members[k, spn]

    def put_state(self, k, arena, home, state):
        """The inverse of state() for chip k: write name -> global array (state()'s layout) into chip k's arena
        (P, ntiles, LANES) and DRAM homes (P, hpages, LANES), fp32 views, in place, so the next launch starts from it
        (a restart). Either array may be None to write only the other, and the homes may be Pages that hold every
        page of a state register. A register the lowering kept as a constant
        must already hold the value: a state that changes one cannot be loaded into these images. Contiguous per-core
        layouts gather and copy in core chunks across _READERS, touching only cores held by this part."""
        from ttfluid.resident.machine import DBASE
        P = int(self.s["P"])
        all_cores = np.arange(P, dtype=np.int32)
        home_cores = np.flatnonzero((home.row >= 0).any(axis=1)).astype(np.int32) if isinstance(home, Pages) else all_cores

        fields = []
        spaces = set()
        for name, shape, regs in self.shared["state"]:
            v = np.asarray(state[name], np.float32)
            v = v.reshape(v.shape[0], -1) if shape else v[:, None]
            assert v.shape[1] == len(regs), (name, v.shape, len(regs))
            reg_ops = []
            for i, (spn, t, n) in enumerate(regs):
                if t is None:
                    c = np.asarray(self.shared["const_state"][f"{name}/{i}"], np.float32)
                    assert np.array_equal(np.broadcast_to(c, v[:, i].shape), v[:, i]), \
                        (f"{name}/{i} is a constant in these images and the state changes it", float(np.max(np.abs(c - v[:, i]))))
                    continue
                X, tt = (home, t - DBASE) if t >= DBASE else (arena, t)
                if X is not None:
                    reg_ops.append((i, spn, X, tt))
                    spaces.add(spn)
            if reg_ops:
                fields.append((v, reg_ops))
        if not fields:
            return
        self._warm_layouts([(k, spn) for spn in spaces])

        def put_general(X, idx, col):
            X[idx] = col

        def put_chunk(v, reg_ops, ptr, ids_s, chunk):
            for c in chunk:
                c = int(c)
                a0, a1 = int(ptr[c]), int(ptr[c + 1])
                if a0 == a1:
                    continue
                nc = a1 - a0
                ntc = -(-nc // LANES)
                active = []
                for i, _, X, tt in reg_ops:
                    if isinstance(X, Pages) and not (X.row[c, tt:tt + ntc] >= 0).any():
                        continue
                    active.append((i, X, tt))
                if not active:
                    continue
                idx_ids = ids_s[a0:a1]
                if len(active) == 1:
                    i, X, tt = active[0]
                    _write_core(X, c, tt, ntc, nc, v[idx_ids, i])
                else:
                    vc = v[idx_ids]
                    for i, X, tt in active:
                        _write_core(X, c, tt, ntc, nc, vc[:, i])

        nw = max(1, _READERS._max_workers)
        jobs = []
        for v, reg_ops in fields:
            spn0 = reg_ops[0][1]
            same_sp = all(spn == spn0 for _, spn, _, _ in reg_ops)
            lay = self._layout(k, spn0) if same_sp else None
            if lay is not None and lay[0]:
                _, ptr, ids_s = lay
                cores = all_cores if any(X is arena for _, _, X, _ in reg_ops) else home_cores
                if len(cores) == 0:
                    continue
                step = max(1, -(-len(cores) // nw))
                for p0 in range(0, len(cores), step):
                    jobs.append(_READERS.submit(put_chunk, v, reg_ops, ptr, ids_s, cores[p0:p0 + step]))
            else:
                g = {}
                for i, spn, X, tt in reg_ops:
                    core, tile, lane, ids = self.members(k, spn)
                    if spn not in g:
                        g[spn] = v[ids]
                    jobs.append(_READERS.submit(put_general, X, (core, tt + tile, lane), g[spn][:, i]))
        for j in jobs:
            j.result()


PAGE = 4 * LANES                           # bytes of a DRAM page: one row of the constants or the homes
GAP, CHUNK, IOV = 1 << 18, 1 << 26, 1024   # read_into: stored ranges closer than GAP share a read of at most CHUNK bytes
_SCRATCH = memoryview(bytearray(GAP))      # where the zeros between them go (never read; shared by all readers)
_READERS = ThreadPoolExecutor(int(os.environ.get("CFD_IMG_THREADS", 8)))   # preadv and numpy indexing release the GIL


class _ChipFiles(dict):
    """A chip directory read lazily: each array is mapped when it is first asked for. Mapping all ~27 at once kept one
    file descriptor per array alive, and 32 chips uploading in parallel ran past `ulimit -n 1024` (EMFILE)."""

    def __init__(self, d):
        super().__init__((f.stem.replace("@", "/"), f) for f in sorted(d.glob("*.npy")))

    def __getitem__(self, name):
        return np.load(dict.__getitem__(self, name), mmap_mode="r")

    def get(self, name, default=None):
        return self[name] if name in self else default

    def stored(self, name):
        """(off, end, ranges): the file offsets where array `name` starts and ends, and the byte ranges
        [(lo, hi), ...] its file stores. write_chip left the zero pages as holes."""
        return _stored(str(dict.__getitem__(self, name)))

    def pages(self, name):
        """One bool per PAGE bytes of array `name`: the file stores some of the page. A page that is False is zero."""
        off, end, ranges = self.stored(name)
        r = np.array(ranges, np.int64).reshape(-1, 2) - off
        step = np.zeros(-(-(end - off) // PAGE) + 1, np.int32)
        np.add.at(step, r[:, 0] // PAGE, 1)
        np.add.at(step, -(-r[:, 1] // PAGE), -1)
        return np.cumsum(step[:-1]) > 0

    def read_into(self, name, out, row=None):
        """Array `name` read into out, a zeroed C-contiguous buffer, from the byte ranges the file stores. A hole is
        not read, so its pages of out stay untouched: they take no time and no RAM (Small k2: 5.3 of 27.3 GB is
        stored). row: out holds a run of the pages pages() marks, page p in row row[p] and -1 for a page outside it
        (a staged upload, whole or in parts); without it out is the whole array. The zero rows of the constants sit between data every ~60 KB and one read per stored range
        is slow, so ranges closer than GAP share one vectored read whose zeros between them land in a scratch buffer,
        not in out."""
        buf = memoryview(out).cast("B")
        off, end, ranges = self.stored(name)
        with open(dict.__getitem__(self, name), "rb") as f:
            fd = f.fileno()
            assert end - off == (len(buf) if row is None else len(row) * PAGE), (name, end - off, len(buf))
            # where the byte at file offset x goes: a stored range is whole pages in a row, so it stays one piece
            at = (lambda x: x - off) if row is None else (lambda x: int(row[(x - off) // PAGE]) * PAGE + (x - off) % PAGE)
            held = None if row is None else np.flatnonzero(row >= 0)
            lo_b, hi_b = (off, end) if held is None else (off + int(held[0]) * PAGE, off + (int(held[-1]) + 1) * PAGE) if len(held) else (off, off)
            reads, last = [], None      # a read: [file offset, bytes, [memoryview, ...]], contiguous in the file
            for a, p in ranges:
                a, p = max(a, lo_b), min(p, hi_b)      # the part of the range inside the run of pages out holds
                for lo in range(a, p, CHUNK):
                    hi = min(lo + CHUNK, p)
                    r = reads[-1] if reads else None
                    piece = buf[at(lo):at(lo) + hi - lo]
                    if r is not None and lo - last <= GAP and len(r[2]) + 2 <= IOV and r[1] + hi - last <= CHUNK:
                        if lo > last:
                            r[2].append(_SCRATCH[:lo - last])
                        r[2].append(piece)
                        r[1] += hi - last
                    else:
                        reads.append([lo, hi - lo, [piece]])
                    last = hi

            def read(lo, size, views):
                n = os.preadv(fd, views, lo)
                assert n == size, (name, lo, size, n)
            list(_READERS.map(lambda r: read(*r), reads))


@functools.lru_cache(None)
def _stored(path):
    """_ChipFiles.stored for one file, walked once: a Small k2 constants file has ~30,000 stored ranges."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0)(f)
        off, fd = f.tell(), f.fileno()
        end = os.fstat(fd).st_size
        ranges, p = [], off
        while p < end:
            try:
                a = os.lseek(fd, p, os.SEEK_DATA)
            except OSError:      # ENXIO: a hole to the end of the file
                break
            p = os.lseek(fd, a, os.SEEK_HOLE)
            ranges.append((a, p))
    return off, end, ranges


def _npy_head(arr):
    f = io.BytesIO()
    np.lib.format.write_array_header_1_0(f, np.lib.format.header_data_from_array_1_0(arr))
    assert f.tell() <= PAGE
    return f.getvalue()


def _used_pages(arr):
    """Which 4-KB blocks of np.save's file of arr hold a non-zero byte, the first (it has the header) left out."""
    b = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)[PAGE - len(_npy_head(arr)):]
    n = len(b) // PAGE
    return np.r_[b[:n * PAGE].reshape(n, PAGE).any(1), b[n * PAGE:].any() if len(b) > n * PAGE else np.zeros(0, bool)]


def _save_sparse(f, arr, used=None):
    """np.save(f, arr) with the file's zero blocks left as holes."""
    arr = np.ascontiguousarray(arr)
    b = arr.reshape(-1).view(np.uint8)
    used = _used_pages(arr) if used is None else used
    edge = np.flatnonzero(np.diff(np.r_[False, used, False]))   # runs of used blocks: [start, end) pairs
    head = _npy_head(arr)
    with open(f, "wb") as fp:
        fp.write(head)
        fp.write(b[:PAGE - len(head)])
        for lo, hi in zip(edge[::2].tolist(), edge[1::2].tolist()):
            fp.seek((1 + lo) * PAGE)
            fp.write(b[lo * PAGE + PAGE - len(head):hi * PAGE + PAGE - len(head)])
        fp.truncate(len(head) + len(b))


def check_free(d, need, floor_gb):
    """Refuse a write of `need` bytes under d when it would leave the disk with less than floor_gb GB free: a full root
    on a shared host kills every other job on it."""
    a = Path(d).absolute()
    while not a.exists():
        a = a.parent
    free = shutil.disk_usage(a).free
    if free - need < floor_gb * 1e9:
        raise SystemExit("refusing to write %.1f GB under %s: %.1f GB free, the floor is %d GB" % (need / 1e9, d, free / 1e9, floor_gb))
