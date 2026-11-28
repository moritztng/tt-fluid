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
import hashlib
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LANES = 1024
PAGE = 4096
SCALARS = ("K", "P", "T", "ndram", "hpages", "cpages", "ntiles", "nhalo", "hmax", "red2", "redm", "L0", "TBP", "tab_stream", "nmaxop",
           "loop_pages")


class Image:
    def __init__(self, shared, chips):
        self.shared = shared          # name -> array or JSON-able object
        self._chips = chips           # k -> dict, or k -> path (lazy)
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
    def chip_of(E, tab, rows, L0, shared, members):
        """One chip's arrays: E's rows `rows` (a slice over E's cores), tab its rewritten table rows (lists, indexed
        like E's), members(space) -> (global ids, core on the chip, lane) of the elements the chip owns."""
        TBP = shared["scalars"]["TBP"]
        t = np.zeros((rows.stop - rows.start, L0), np.uint32)
        for i, row in enumerate(tab[rows]):
            t[i, :len(row)] = row
        ch = dict(arena=np.ascontiguousarray(E.arena[rows]), home=np.ascontiguousarray(E.home[rows]),
                  consts=np.ascontiguousarray(E.consts[rows]), tab=t,
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
    def write(self, d, floor_gb=0):
        """One pass finds every array's non-zero pages (the disk check needs their count), the write stores only those:
        the files are np.save's byte for byte, with the zero pages as holes (home 97 %, consts 58 % on Small k4)."""
        K = self.s["K"]
        with ThreadPoolExecutor(min(K, 8)) as ex:
            used = dict(kv for kvs in ex.map(lambda k: [((k, n), _used_pages(v)) for n, v in self.chip(k).items()], range(K)) for kv in kvs)
            check_free(d, sum(1 + int(u.sum()) for u in used.values()) * PAGE, floor_gb)
            self.write_shared(d)
            list(ex.map(lambda k: self.write_chip(d, k, used), range(K)))

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
        h = hashlib.sha256()
        for name in ("c", "b", "n"):
            h.update(name.encode() + np.ascontiguousarray(self.shared[name]).tobytes())
        h.update(json.dumps([self.s, self.shared["gtab"], self.shared["out_names"]], sort_keys=True).encode())
        ch = self.chip(k)
        for name in sorted(ch):
            a = np.ascontiguousarray(ch[name])
            h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.reshape(-1).view(np.uint8))   # the array's bytes without the copy tobytes() makes (home is 8 GB a chip on Small)
        return h.hexdigest()

    # ---- results, chip by chip ------------------------------------------------------------------------------------
    def state(self, out_rows, home_rows):
        """out_rows(k) -> chip k's arena (P, ntiles, LANES) fp32; home_rows(k) -> its DRAM homes (P, hpages, LANES).
        Returns name -> global array, as Encoded.state_from_arena."""
        P, K = self.s["P"], self.s["K"]
        sp_n = self.shared["spaces"]
        out = {}
        regs_all = [(name, shape, regs) for name, shape, regs in self.shared["state"]]
        glob = {}
        for name, shape, regs in regs_all:
            for i, (spn, t, n) in enumerate(regs):
                if t is None:
                    glob[(name, i)] = np.asarray(self.shared["const_state"][f"{name}/{i}"], np.float32)
                else:
                    glob[(name, i)] = np.zeros(sp_n[spn], np.float32)
        for k in range(K):
            ch = self.chip(k)
            A, H = out_rows(k), home_rows(k)
            for name, shape, regs in regs_all:
                for i, (spn, t, n) in enumerate(regs):
                    if t is None:
                        continue
                    from ttfluid.resident.machine import DBASE
                    X, tt = (H, t - DBASE) if t >= DBASE else (A, t)
                    flat = X[:, tt:tt + n].reshape(P, -1)
                    ids = ch["ids/" + spn]
                    glob[(name, i)][ids] = flat[ch["core/" + spn], ch["lane/" + spn]]
        for name, shape, regs in regs_all:
            arrs = [glob[(name, i)] for i in range(len(regs))]
            out[name] = np.stack(arrs, -1).reshape(arrs[0].shape + tuple(shape)) if shape else arrs[0]
        return out

    def put_state(self, k, arena, home, state):
        """The inverse of state() for chip k: write name -> global array (state()'s layout) into chip k's arena
        (P, ntiles, LANES) and DRAM homes (P, hpages, LANES), fp32 views, in place, so the next launch starts from it
        (a restart). Either array may be None to write only the other. A register the lowering kept as a constant
        must already hold the value: a state that changes one cannot be loaded into these images."""
        from ttfluid.resident.machine import DBASE
        for name, shape, regs in self.shared["state"]:
            v = np.asarray(state[name], np.float32)
            v = v.reshape(v.shape[0], -1) if shape else v[:, None]
            assert v.shape[1] == len(regs), (name, v.shape, len(regs))
            for i, (spn, t, n) in enumerate(regs):
                if t is None:
                    c = np.asarray(self.shared["const_state"][f"{name}/{i}"], np.float32)
                    assert np.array_equal(np.broadcast_to(c, v[:, i].shape), v[:, i]), \
                        (f"{name}/{i} is a constant in these images and the state changes it", float(np.max(np.abs(c - v[:, i]))))
                    continue
                X, tt = (home, t - DBASE) if t >= DBASE else (arena, t)
                if X is None:
                    continue
                ch = self.chip(k)
                lane = np.asarray(ch["lane/" + spn])
                X[np.asarray(ch["core/" + spn]), tt + lane // LANES, lane % LANES] = v[np.asarray(ch["ids/" + spn]), i]


class _ChipFiles(dict):
    """A chip directory read lazily: each array is mapped when it is first asked for. Mapping all ~27 at once kept one
    file descriptor per array alive, and 32 chips uploading in parallel ran past `ulimit -n 1024` (EMFILE)."""

    def __init__(self, d):
        super().__init__((f.stem.replace("@", "/"), f) for f in sorted(d.glob("*.npy")))

    def __getitem__(self, name):
        return np.load(dict.__getitem__(self, name), mmap_mode="r")

    def get(self, name, default=None):
        return self[name] if name in self else default

    def values(self):
        return [self[n] for n in self]

    def items(self):
        return [(n, self[n]) for n in self]


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
