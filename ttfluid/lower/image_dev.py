"""run_mesh.py --from-images DIR: the device run from per-chip images (lower/image.py) instead of a lowering, uploading and
downloading chip by chip so the host never holds all K chips' arrays (an occDrivAer 65M image is ~6.3 GB per chip).

    IMG, m, E, gtab, info, cells, place, SHS, NSH = setup(d, LOWER, K, P, R, C)
    tab_gw, chip_tab, L = gateway_tables(IMG, gtab, xyw, K, P)
    t = upload(md, K, DEVBLK, shape, fill)       # fill(k, out) writes chip k's (rows, words) uint32 array into out
    x = download_chip(t, d)                      # device d's shard of a mesh tensor, as numpy

m and E are stand-ins carrying exactly the attributes run_mesh's device code reads. Upload: a mesh tensor is allocated
on the device and written from one host buffer that holds only the bytes the image files store (host_shards). The
constants and the homes are mostly zero pages: the host writes only the pages that hold a value and the chips lay the
arrays themselves (upload_placed). CFD_IMG_VERIFY=1 reads every uploaded tensor back chip by chip and compares it
with the image.
"""
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttfluid.lower.image import Image, LANES, Pages, _READERS, par_zero
from ttfluid.lower.image import load_state as load_state

KERNELS = Path(__file__).resolve().parent.parent / "device" / "kernels"


def set_outer(c, n_old, n_new):
    """The compute stream with the outer-iteration cap n_old replaced by n_new. program.build ends every pass with
    `jumpif IT < max_outer`, encoded as C_FILL slot <- max_outer, C_SUB d <- IT - slot, C_UN ltz d: the cap is that
    fill's immediate and nothing else reads it, so images lowered at one n run any other n in one launch (the same
    passes as a chain of launches: IT is not state, every launch counts from 0). Lowering at another n is not an
    option for this: it renumbers registers."""
    from ttfluid.resident.encode import C_FILL, C_BIN, C_UN
    c = np.array(c, np.uint32)
    imm = np.float32(n_old).view(np.uint32)
    hit = [i for i in np.nonzero(c[1:-2] == imm)[0] + 1
           if c[i - 1] & 0xFF == C_FILL and c[i + 1] & 0xFF == C_BIN["sub"] and c[i + 2] & 0xFF == C_UN["ltz"]
           and (c[i + 1] >> 16) & 0xF == (c[i - 1] >> 8) & 0xF and (c[i + 2] >> 8) & 0xF == (c[i + 1] >> 8) & 0xF]
    assert len(hit) == 1, ("outer-iteration cap %d: %d sites in the compute stream" % (n_old, len(hit)), hit)
    c[hit[0]] = np.float32(n_new).view(np.uint32)
    print("outer-iteration cap %d -> %d (compute word %d)" % (n_old, n_new, hit[0]), flush=True)
    return c


def flag_value(k, v):
    """A lowering flag as the images are matched on. --forces "CONTROLDICT:FUNC" names a file of the case, and the case may
    sit elsewhere on the device host (another home, a relative path): the lowering baked the coefficients in, so only the
    file name and the function identify it."""
    if k == "forces" and v and ":" in v:
        path, func = v.rsplit(":", 1)
        return os.path.basename(path) + ":" + func
    return v


def setup(d, LOWER, K, P, R, C):
    img = Image.load(d)
    s, sh = img.s, img.shared
    flags = dict(sh["flags"])
    # the case directory may sit elsewhere on the device host: the images carry everything the run reads from it
    diff = {k: (flags.get(k), LOWER[k]) for k in LOWER
            if k not in ("case", "n") and flag_value(k, flags.get(k)) != flag_value(k, LOWER[k])}
    # --phase: the images' markers are in the code, so images lowered with it run with it (run_mesh turns it on);
    # --phase on images without markers is refused above (flags phase None vs True)
    assert not diff, ("--from-images: lowering flags differ from the images'", diff)
    assert s["K"] == K and s["P"] == P, (s["K"], s["P"], K, P)
    info = sh["info"]
    assert "shs" in info, "images written before the shadow size was recorded (info shs/nsh): lower again"
    assert "cpages" in s, "images written before packed DRAM constants/homes (page core * pages + offset + t): lower again"
    m = SimpleNamespace(ntiles=s["ntiles"], ndram=s["ndram"], hpages=s["hpages"], cpages=s["cpages"], T=s["T"], red2=s["red2"], redm=s["redm"], nhalo=s["nhalo"],
                        maps=SimpleNamespace(hmax=s["hmax"], plans=[None] * s.get("nplans", 1)),
                        hwords=s.get("hwords", max(s["nhalo"] * s["hmax"], 1)))
    tab_halo = np.concatenate([np.asarray(img.chip(k)["tab_halo"]) for k in range(K)])
    sizes = dict(sh["sizes"])
    c = sh["c"] if flags.get("n") == LOWER["n"] else set_outer(sh["c"], flags["n"], LOWER["n"])
    # tab_stream as the encoding has it where run_mesh reads it: the local entry counts of --bpatch auto
    E = SimpleNamespace(c=c, b=sh["b"], n=sh["n"], tab_stream=dict(loc=info["loc"]) if "loc" in info else s["tab_stream"], loop_pages=s["loop_pages"],
                        n_maxop=lambda: s["nmaxop"], sizes=lambda: sizes, out_names=list(sh["out_names"]),
                        tab_halo=tab_halo)
    gtab = {int(g): v for g, v in sh["gtab"].items()}
    cells = [(r, c) for r in range(R) for c in range(C)]
    place = [cells.index(tuple(info["place"][str(k)] if str(k) in info["place"] else info["place"][k])) for k in range(K)]
    return img, m, E, gtab, info, cells, place, int(info["shs"]), int(info["nsh"])


def gw_words(v, xyw, P):
    w = [v["gR"], v["gP"], xyw[v["peer"] % P], len(v["rec"])]
    for (i, q, lo, n, cnt) in v["rec"]:
        w += [xyw[i % P], q, lo, n, cnt]
    w += [len(v["mem"])]
    for (j, k) in v["mem"]:
        w += [j % P, k]
    return w


def gateway_tables(img, gtab, xyw, K, P):
    """As run_mesh's global block: every row padded to L0, a gateway's own words after it; rows padded to the widest
    (a whole number of 16 words). chip_tab(k) -> chip k's (P, L) uint32 table."""
    L0 = img.s["L0"]
    tab_gw = [0] * (K * P)
    gw = {g: gw_words(v, xyw, P) for g, v in gtab.items()}
    for g in gw:
        tab_gw[g] = L0
    L = max([L0] + [L0 + len(w) for w in gw.values()])
    L = -(-L // 16) * 16

    def chip_tab(k):
        t = np.zeros((P, L), np.uint32)
        t[:, :L0] = img.chip(k)["tab"]
        for g, w in gw.items():
            if g // P == k:
                t[g % P, L0:L0 + len(w)] = w
        return t
    return tab_gw, chip_tab, L


def pad16(arr):
    arr = np.ascontiguousarray(np.asarray(arr).view(np.uint32))
    w = -(-arr.shape[1] // 16) * 16
    if w == arr.shape[1]:
        return arr
    out = np.zeros((arr.shape[0], w), np.uint32)
    out[:, :arr.shape[1]] = arr
    return out


def chip_shape(img, P, name):
    """(rows, words) of one chip's device array `name` (arena, consts, home, blob): the same on every chip."""
    if name == "blob":
        return P * img.s["TBP"], LANES
    n = img.chip(0)[name].size       # mapped: only the header is read
    return (P, -(-n // P // 16) * 16) if name == "arena" else (n // LANES, LANES)


def chip_fill(img, k, P, name, out, state=None, row=None):
    """Chip k's device array `name` as run_mesh's device code builds it from E, written into out: a zeroed
    (rows, words) uint32 buffer of chip_shape. Only the bytes the image file stores are read, so the zero pages of
    the constants and the homes (80 % of a Small image) are never touched on the host. state: name -> global array
    (Image.state's layout) written over the arena and the homes, so the run starts from it (run_mesh --state-in).
    row: out holds only the pages chip_pages marks, page p in row row[p] (Staged)."""
    ch = img.chip(k)
    if name == "arena":
        a = np.asarray(ch["arena"]).reshape(P, -1).view(np.uint32)
        out[:, :a.shape[1]] = a
    elif name != "blob" or img.s["tab_stream"]:
        ch.read_into(name, out, row)
    if state is not None and name in ("arena", "home"):
        f = out.view(np.float32)
        nt = img.s["ntiles"]
        img.put_state(k, f[:, :nt * LANES].reshape(P, nt, LANES) if name == "arena" else None,
                      (f.reshape(P, -1, LANES) if row is None else Pages(f, row.reshape(P, -1))) if name == "home" else None, state)


def chip_pages(img, k, P, name, state=None):
    """One bool per page of chip k's constants or homes: the page holds a value, because the image file stores it or
    because the state is laid over it. Every other page is zero."""
    on = img.chip(k).pages(name)
    if state is not None and name == "home":
        for lo, hi in state_pages(img):
            on.reshape(P, -1)[:, lo:hi] = True
    return on


class Staged:
    """An array of 4 KB pages as upload_placed sends it: only the pages that hold a value, in page order, in one part
    or several. mark() reads the page marks of every chip (on[k], pages(k) on a thread per chip) and decides: sparse
    when the host would write less than half of the array, and as many equal runs of pages (parts) as it takes for
    the staged pages of one run, over all chips, to fit in cap bytes, so that the host holds parts and never the
    array (Ahead). part(j): its first page, the pages of a full part, its staging shape and fill, and on(k), chip
    k's marks inside it. pieces(): what the host writes, in order: the parts when sparse, else the array itself."""

    def __init__(self, DEVBLK, shape, fill, pages, cap=None):
        self.DEVBLK, self.full, self.fill_full, self.pages, self.cap, self.on = DEVBLK, shape, fill, pages, cap, None
        self.lock = threading.Lock()

    def mark(self):
        with self.lock:
            if self.on is None:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(len(self.DEVBLK)) as ex:
                    self.on = dict(zip(self.DEVBLK, ex.map(self.pages, self.DEVBLK)))
                c = [np.concatenate([[0], np.cumsum(o)]) for o in self.on.values()]
                self.shape = (max(1, max(int(x[-1]) for x in c)), self.full[1])
                self.sparse = 2 * self.shape[0] < self.full[0]
                n = 1
                while True:
                    self.size = -(-self.full[0] // n)
                    e = np.minimum(np.arange(n + 1) * self.size, self.full[0])
                    self.rows = np.maximum(1, np.max([x[e[1:]] - x[e[:-1]] for x in c], axis=0))
                    if self.cap is None or not self.sparse or n == 64 or int(self.rows.max()) * len(c) * 4 * self.full[1] <= self.cap:
                        break
                    n += 1
                self.parts = n
        return self

    def part(self, j):
        a, b = j * self.size, min((j + 1) * self.size, self.full[0])

        def fill(k, out):
            row = np.full(self.full[0], -1, np.int64)
            row[a:b] = np.where(self.on[k][a:b], np.cumsum(self.on[k][a:b]) - 1, -1)
            self.fill_full(k, out, row)
        return SimpleNamespace(first=a, size=self.size, shape=(int(self.rows[j]), self.full[1]), fill=fill, on=lambda k: self.on[k][a:b])

    def pieces(self):
        self.mark()
        return [self.part(j) for j in range(self.parts)] if self.sparse else [SimpleNamespace(shape=self.full, fill=self.fill_full)]


def _bg_drop(items):
    """Release large numpy buffers off the critical path so munmap TLB shootdowns do not stall upload."""
    if items:
        threading.Thread(target=lambda x: x.clear(), args=(list(items),), daemon=True).start()


class Ahead:
    """The host buffers of a Staged array's pieces, built in order on a thread of their own from the moment this is
    made: a launch makes it before ttnn is loaded. next() hands out the next buffer, done() returns it to the pool
    once written and synchronized. `alive` buffers at most are built and not yet done, and multi-part uploads reuse
    those pooled buffers with parallel zeroing instead of reallocating and unmapping every part."""

    def __init__(self, sg, alive=2):
        self.built, self._free = queue.Queue(), queue.Queue()
        self._in_flight, self._pool, self._done, self._total = None, [], 0, None
        for _ in range(alive):
            self._free.put(None)

        def build():
            try:
                pieces = sg.pieces()
                self._total = len(pieces)
                reuse = sg.sparse and len(pieces) > 1
                K, max_rows, words = len(sg.DEVBLK), int(sg.rows.max()) if reuse else 0, sg.full[1]
                for p in pieces:
                    slot = self._free.get()
                    if reuse:
                        if slot is None:
                            slot = np.empty((K * max_rows, words), np.uint32)
                            self._pool.append(slot)
                        view = host_buffer(K, sg.DEVBLK, p.shape, p.fill, buf=slot)
                    else:
                        view = host_buffer(K, sg.DEVBLK, p.shape, p.fill)
                        slot = view
                        self._pool.append(slot)
                    self.built.put((slot, view))
            except BaseException as e:     # next() raises it where the buffer was expected
                self.built.put(e)
        threading.Thread(target=build, daemon=True).start()

    def next(self):
        item = self.built.get()
        if isinstance(item, BaseException):
            raise item
        slot, view = item
        self._in_flight = slot
        return view

    def done(self):
        slot, self._in_flight = self._in_flight, None
        self._done += 1
        if self._total is not None and self._done >= self._total:
            while not self._free.empty():
                try:
                    self._free.get_nowait()
                except queue.Empty:
                    break
            pool, self._pool = self._pool, []
            _bg_drop(pool)
        elif slot is not None:
            self._free.put(slot)


def page_map(on, W, size=None):
    """A run of an array's pages split over W workers: (map, args) from one bool per page. Worker w moves pages
    first .. first + count - 1 of the run; bit j of its map row (uint32 words) says page first + j is staged, and
    its staged pages start at row start of the staging tensor. args[w] = (w, first, count, start), what
    kernels/place.cpp takes. size: lay the map out for a run of this many pages (the last part of a staged upload is
    shorter than the others and shares their kernel build)."""
    n = -(-(size or len(on)) // W)
    bits = np.zeros((W, -(-n // 512) * 512), bool)      # a map row is whole 64-byte DRAM reads
    pad = np.zeros(W * n, bool)
    pad[:len(on)] = on
    bits[:, :n] = pad.reshape(W, n)
    start = np.concatenate([[0], np.cumsum(pad)])[::n]
    args = [(w, w * n, max(0, min(n, len(on) - w * n)), int(start[w])) for w in range(W)]
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32), args


def map_pages(m, rows):
    """page_map's inverse: one bool per page from the (W, words) map of an array of `rows` pages."""
    n = -(-rows // len(m))
    return np.unpackbits(np.ascontiguousarray(m).view(np.uint8), axis=1, bitorder="little")[:, :n].reshape(-1)[:rows].astype(bool)


def carry_arena(img, k, P, a_out):
    """Chip k's arena for the next leg of an in-process chain (run_mesh --legs): the image's arena with every state
    register the last leg left in L1 copied over from its dump a_out (P, ntiles, LANES), at exactly the lanes
    put_state writes and state() reads, so the leg starts where a --state-in restart from that state would."""
    from ttfluid.resident.machine import DBASE
    ch = img.chip(k)
    nt = img.s["ntiles"]
    ar = pad16(np.asarray(ch["arena"]).reshape(P, -1)).copy()
    A = ar[:, :nt * LANES].view(np.float32).reshape(P, nt, LANES)
    src = np.asarray(a_out).view(np.float32)
    for name, shape, regs in img.shared["state"]:
        for spn, t, n in regs:
            if t is None or t >= DBASE:
                continue
            core, lane = np.asarray(ch["core/" + spn]), np.asarray(ch["lane/" + spn])
            idx = (core, t + lane // LANES, lane % LANES)
            A[idx] = src[idx]
    return ar


def upload(md, K, DEVBLK, shape, fill, name=None, host=None, out=None):
    """A (K * rows, words) uint32 mesh tensor whose device d holds the array fill(DEVBLK[d], out) writes into its zeroed
    (rows, words) buffer out, all shards written in one copy. host: a call that gives host_shards(...) of this array,
    built ahead (a future's result, or a buffer of Ahead). out: reuse an existing device tensor in place. With
    CFD_UPLOAD_PROFILE set and a name, prints where the upload's wall clock went: waiting for the host buffer (image
    read, state overlay) against the write to the chips, which is synchronized so the split is the device's."""
    import ttnn
    prof = name and os.environ.get("CFD_UPLOAD_PROFILE")
    t0 = time.time()
    box = host() if host is not None else host_shards(md, K, DEVBLK, shape, fill)
    h, shard = box
    box.clear()   # a future keeps its result: without this the host buffer would outlive the write
    t1 = time.time()
    t = out if out is not None else ttnn.allocate_tensor_on_device(
        shard, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy_host_to_device_tensor(h, t)
    if prof:
        ttnn.synchronize_device(md)
        t2 = time.time()
    del h
    if prof:
        mb = shape[0] * shape[1] * 4 / 2**20
        print("upload %s: %d chips x %.0f MB, host shards %.2f s, write %.2f s (%.0f MB/s over the mesh), free %.2f s"
              % (name, K, mb, t1 - t0, t2 - t1, K * mb / max(t2 - t1, 1e-9), time.time() - t2), flush=True)
    verify(t, K, DEVBLK, shape, fill, name)
    return t


def verify(t, K, DEVBLK, shape, fill, name):
    """CFD_IMG_VERIFY: every chip's shard of t read back and compared with the array fill builds."""
    if os.environ.get("CFD_IMG_VERIFY"):
        for d in range(K):
            ref = np.zeros(shape, np.uint32)
            fill(DEVBLK[d], ref)
            assert np.array_equal(download_chip(t, d), ref), ("upload read back differs", name, d)


def workers(md, cells, grid, cores, DEVBLK, source, ct, tensors, out, common, args, l1):
    """Runs kernels/<source> on both data movement processors of every core of every chip: 2 * cores workers a chip.
    Worker w (core w // 2, processor w % 2) on the chip of block k gets the runtime args args(k)[w] and an L1 buffer
    of l1 bytes; the compile-time args are [that buffer] + ct + the accessors of `tensors`, the common args the
    tensors' addresses + common. out: the tensor the program writes. cells[d]: device d's mesh coordinate."""
    import ttnn
    acc = [a for x in tensors for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]
    mpd = ttnn.MeshProgramDescriptor()
    for d, rc in enumerate(cells):
        rows = args(DEVBLK[d])
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [ttnn.CBDescriptor(total_size=l1, core_ranges=grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=l1)]) for i in (0, 1)]
        kernels = []
        for i, (proc, noc) in enumerate(((ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                                         (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1))):
            kd = ttnn.KernelDescriptor(kernel_source=str(KERNELS / source), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                       core_ranges=grid, compile_time_args=[i] + list(ct) + acc,
                                       config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc))
            rt = ttnn.RuntimeArgs()
            for c, (x, y) in enumerate(cores):
                rt[x][y] = list(rows[2 * c + i])
            kd.runtime_args = rt
            kd.common_runtime_args = [x.buffer_address() for x in tensors] + list(common)
            kernels.append(kd)
        prog.kernels = kernels
        mc = ttnn.MeshCoordinate(*rc)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
    ttnn.generic_op([x for x in tensors if x is not out] + [out], mpd)
    ttnn.synchronize_device(md)


NB = 16   # pages a worker keeps in flight


def upload_placed(md, cells, grid, cores, DEVBLK, sg, name=None, ahead=None, out=None):
    """upload for a Staged array. Sparse: part by part the host writes the staged pages and a page map, and every
    core of a chip lays its share of the part with both its data movement processors (kernels/place.cpp), copying the
    staged pages to their places and writing the rest as zeros. The array ends up as upload leaves it; the PCIe write
    carries a fifth of the bytes (zeros cost it what data costs) and the host never holds the zero pages. Not
    sparse: a plain upload. cells[d]: device d's mesh coordinate; ahead: the Ahead of sg, started earlier; out: reuse
    an existing device tensor in place (keeping its buffer address unchanged across --cases / --runs)."""
    import torch
    import ttnn
    K, W = len(DEVBLK), 2 * len(cores)
    prof = name and os.environ.get("CFD_UPLOAD_PROFILE")
    ahead = ahead or Ahead(sg)
    parts = sg.pieces()

    def write(p, label, dst=None):
        return upload(md, K, DEVBLK, p.shape, p.fill, label,
                      lambda: host_shards(md, K, DEVBLK, p.shape, p.fill, ahead.next()), out=dst)
    if not sg.sparse:
        t = write(parts[0], name, dst=out)
        if not prof:
            ttnn.synchronize_device(md)
        ahead.done()
        return t
    t = out if out is not None else ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(sg.full)), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    t0 = time.time()
    # Precompute page maps for all parts (parallel across parts) and reuse one t_map tensor across the loop.
    map_jobs = [(j, k, p.on(k), p.size) for j, p in enumerate(parts) for k in DEVBLK]
    if len(map_jobs) > 1:
        map_res = list(_READERS.map(lambda it: page_map(it[2], W, it[3]), map_jobs))
    else:
        map_res = [page_map(it[2], W, it[3]) for it in map_jobs]
    works = [{} for _ in parts]
    for (j, k, _, _), res in zip(map_jobs, map_res):
        works[j][k] = res
    mw = works[0][DEVBLK[0]][0].shape[1]
    map_buf = np.empty((K * W, mw), np.uint32)
    t_map = ttnn.allocate_tensor_on_device(ttnn.Shape([W, mw]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    maps = time.time() - t0
    laid = 0.0
    for j, p in enumerate(parts):
        t_stage = write(p, name and "%s staged %d/%d" % (name, j + 1, len(parts)))
        t0 = time.time()
        work = works[j]
        for d, k in enumerate(DEVBLK):
            map_buf[d * W:(d + 1) * W] = work[k][0]
        h_map = ttnn.from_torch(torch.from_numpy(map_buf.view(np.int32)), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0))
        ttnn.copy_host_to_device_tensor(h_map, t_map)
        del h_map
        t1 = time.time()
        workers(md, cells, grid, cores, DEVBLK, "place.cpp", [4 * mw, NB, 0], [t, t_stage, t_map], t, [0],
                lambda k, w_=work, pf=p.first: [(w, pf + first, count, start) for w, first, count, start in w_[k][1]],
                4 * mw + 4096 * (NB + 1))
        ttnn.deallocate(t_stage)
        ahead.done()
        maps, laid = maps + t1 - t0, laid + time.time() - t1
    ttnn.deallocate(t_map)
    if prof:
        print("place %s: %d of %d pages per chip staged in %d part(s), maps %.2f s, laid on the chips in %.2f s (compile included)"
              % (name, sg.shape[0], sg.full[0], len(parts), maps, laid), flush=True)
    verify(t, K, DEVBLK, sg.full, sg.fill_full, name)
    return t


def host_buffer(K, DEVBLK, shape, fill, buf=None):
    """The K shards of an array in one zeroed (K * rows, words) uint32 buffer, shard d filled by fill(DEVBLK[d], out)
    on a thread of its own. The zeros cost RAM only where fill writes. When buf is supplied (a pooled buffer from
    Ahead), its first K * rows rows are zeroed in parallel and reused without reallocating. numpy and file reads only:
    a launch builds its arrays with this while ttnn loads and the mesh opens."""
    from concurrent.futures import ThreadPoolExecutor
    rows, words = shape
    if buf is None:
        view = np.zeros((K * rows, words), np.uint32)
    else:
        view = buf[:K * rows]
        par_zero(view)
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(lambda d: fill(DEVBLK[d], view[d * rows:(d + 1) * rows]), range(K)))
    return view


def host_shards(md, K, DEVBLK, shape, fill, buf=None):
    """upload's host side: [the K shards as one multi-device host tensor, one shard's shape], ready for
    copy_host_to_device_tensor (run_mesh --cases builds the next case's homes with it while the device runs this one).
    The shards are slices of one buffer (buf, or host_buffer's) that ttnn borrows: a shard along dim 0 of a row-major
    buffer needs no copy (a from_torch without a mesh mapper always makes one, at ~1 GB/s). One write of all shards
    also sidesteps copy_host_to_device_tensor into a get_device_tensors() view, which writes EVERY device (ttnn 0.68:
    all 8 chips read back the last chip's data). A list, so that upload can take the tensor out of a future's result
    and the buffer is freed when the write is done."""
    import torch
    import ttnn
    buf = host_buffer(K, DEVBLK, shape, fill) if buf is None else buf
    # tagged int32, not uint32: the same bytes, and the program reads the buffer as raw words either way
    h = ttnn.from_torch(torch.from_numpy(buf.view(np.int32)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0))
    return [h, ttnn.Shape([shape[0], shape[1]])]


def download_chip(t, d):
    import ttnn
    return ttnn.to_torch(ttnn.get_device_tensors(t)[d]).numpy().view(np.uint32)


def state_pages(img):
    """The DRAM home page ranges [(lo, hi), ...] (merged, sorted) that hold a state register: all a state download has
    to read (Small k32: 116 of 2,134 pages)."""
    from ttfluid.resident.machine import DBASE
    m = []
    for lo, hi in sorted((t - DBASE, t - DBASE + n) for _, _, regs in img.shared["state"] for _, t, n in regs
                         if t is not None and t >= DBASE):
        if m and lo <= m[-1][1]:
            m[-1][1] = max(m[-1][1], hi)
        else:
            m.append([lo, hi])
    return [tuple(r) for r in m]


def download_state(md, cells, grid, cores, DEVBLK, t, P, HPG, ranges):
    """d -> device d's homes as Pages (uint32) that hold the state pages which are not zero: all the host reads.
    Every worker tests the state pages of its share on the chip and marks them in a page map (kernels/scan.cpp); the
    host reads the maps, and the chips collect the marked pages into one tensor in page order (kernels/place.cpp,
    backwards). After 20 its of Small a third of the state pages hold anything: the unused slots of a cell's face
    row stay zero. ranges: state_pages, inside a core's HPG pages."""
    import ttnn
    K, W, rows = len(DEVBLK), 2 * len(cores), P * HPG
    read = lambda x: ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(np.uint32)
    t0 = time.time()
    m0, geo = page_map(np.zeros(rows, bool), W)
    mw = m0.shape[1]
    alloc = lambda r, w: ttnn.allocate_tensor_on_device(ttnn.Shape([r, w]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    t_map = alloc(W, mw)
    workers(md, cells, grid, cores, DEVBLK, "scan.cpp", [4 * mw, NB, HPG, len(ranges)], [t, t_map], t_map,
            [x for r in ranges for x in r], lambda k: [g[:3] for g in geo], 4 * mw + 4096 * NB)
    t1 = time.time()
    on = [map_pages(m, rows) for m in read(t_map).reshape(K, W, mw)]
    S = max(int(o.sum()) for o in on)
    t_stage = alloc(S + 1, LANES)
    workers(md, cells, grid, cores, DEVBLK, "place.cpp", [4 * mw, NB, 1], [t, t_stage, t_map], t_stage, [S],
            lambda k: page_map(on[DEVBLK.index(k)], W)[1], 4 * mw + 4096 * (NB + 1))
    t2 = time.time()
    x = read(t_stage).reshape(K, S + 1, LANES)
    ttnn.deallocate(t_stage)
    ttnn.deallocate(t_map)
    if os.environ.get("CFD_UPLOAD_PROFILE"):
        print("state download: %d of %d state pages per chip not zero, scan and map %.2f s, collect %.2f s, read %.2f s"
              % (S, P * sum(h - l for l, h in ranges), t1 - t0, t2 - t1, time.time() - t2), flush=True)
    return lambda d: Pages(x[d], np.where(on[d], np.cumsum(on[d]) - 1, S).reshape(P, HPG), zero_row=S)


def spread_pages(parts, P, HPG, ranges):
    """d -> (P, HPG, LANES) uint32 from one (devices * P, hi - lo, LANES) array per page range, in device order."""
    def pages(d):
        out = np.zeros((P, HPG, LANES), np.uint32)
        for (lo, hi), h in zip(ranges, parts):
            out[:, lo:hi] = h[d * P:(d + 1) * P]
        return out
    return pages
