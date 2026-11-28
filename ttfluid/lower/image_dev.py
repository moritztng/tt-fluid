"""run_mesh.py --from-images DIR: the device run from per-chip images (lower/image.py) instead of a lowering, uploading and
downloading chip by chip so the host never holds all K chips' arrays (an occDrivAer 65M image is ~6.3 GB per chip).

    IMG, m, E, gtab, info, cells, place, SHS, NSH = setup(d, LOWER, K, P, R, C)
    tab_gw, chip_tab, L = gateway_tables(IMG, gtab, xyw, K, P)
    t = upload(md, K, DEVBLK, shape, fill)       # fill(k, out) writes chip k's (rows, words) uint32 array into out
    x = download_chip(t, d)                      # device d's shard of a mesh tensor, as numpy

m and E are stand-ins carrying exactly the attributes run_mesh's device code reads. Upload: a mesh tensor is allocated
on the device and written from one host buffer that holds only the bytes the image files store (host_shards);
CFD_IMG_VERIFY=1 reads every uploaded tensor back chip by chip and compares it with the image.
"""
import os
import time
from types import SimpleNamespace

import numpy as np

from ttfluid.lower.image import Image, LANES


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


def chip_fill(img, k, P, name, out, state=None):
    """Chip k's device array `name` as run_mesh's device code builds it from E, written into out: a zeroed
    (rows, words) uint32 buffer of chip_shape. Only the bytes the image file stores are read, so the zero pages of
    the constants and the homes (80 % of a Small image) are never touched on the host. state: name -> global array
    (Image.state's layout) written over the arena and the homes, so the run starts from it (run_mesh --state-in)."""
    ch = img.chip(k)
    if name == "arena":
        a = np.asarray(ch["arena"]).reshape(P, -1).view(np.uint32)
        out[:, :a.shape[1]] = a
    elif name != "blob" or img.s["tab_stream"]:
        ch.read_into(name, out)
    if state is not None and name in ("arena", "home"):
        f = out.view(np.float32)
        nt = img.s["ntiles"]
        img.put_state(k, f[:, :nt * LANES].reshape(P, nt, LANES) if name == "arena" else None,
                      f.reshape(P, -1, LANES) if name == "home" else None, state)


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


def upload(md, K, DEVBLK, shape, fill, name=None, host=None):
    """A (K * rows, words) uint32 mesh tensor whose device d holds the array fill(DEVBLK[d], out) writes into its zeroed
    (rows, words) buffer out, all shards written in one copy. host: a future of host_shards(...) for this array, built
    ahead while another array was written. With CFD_UPLOAD_PROFILE set and a name, prints where the upload's wall clock
    went: waiting for the host buffer (image read, state overlay) against the write to the chips, which is
    synchronized so the split is the device's."""
    import ttnn
    prof = name and os.environ.get("CFD_UPLOAD_PROFILE")
    t0 = time.time()
    h, shard = host.result() if host is not None else host_shards(md, K, DEVBLK, shape, fill)
    t1 = time.time()
    t = ttnn.allocate_tensor_on_device(shard, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy_host_to_device_tensor(h, t)
    if prof:
        ttnn.synchronize_device(md)
        t2 = time.time()
    del h
    if prof:
        mb = shape[0] * shape[1] * 4 / 2**20
        print("upload %s: %d chips x %.0f MB, host shards %.2f s, write %.2f s (%.0f MB/s over the mesh), free %.2f s"
              % (name, K, mb, t1 - t0, t2 - t1, K * mb / max(t2 - t1, 1e-9), time.time() - t2), flush=True)
    if os.environ.get("CFD_IMG_VERIFY"):
        for d in range(K):
            ref = np.zeros(shape, np.uint32)
            fill(DEVBLK[d], ref)
            assert np.array_equal(download_chip(t, d), ref), ("upload read back differs", name, d)
    return t


def host_shards(md, K, DEVBLK, shape, fill):
    """upload's host side: the K shards as one multi-device host tensor and one shard's shape, ready for
    copy_host_to_device_tensor (run_mesh --cases builds the next case's homes with it while the device runs this one).
    The shards are slices of one zeroed buffer, which costs RAM only where fill writes, and ttnn borrows it: a shard
    along dim 0 of a row-major buffer needs no copy (a from_torch without a mesh mapper always makes one, at ~1 GB/s).
    One write of all shards also sidesteps copy_host_to_device_tensor into a get_device_tensors() view, which writes
    EVERY device (ttnn 0.68: all 8 chips read back the last chip's data)."""
    import torch
    import ttnn
    from concurrent.futures import ThreadPoolExecutor
    rows, words = shape
    buf = np.zeros((K * rows, words), np.uint32)
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(lambda d: fill(DEVBLK[d], buf[d * rows:(d + 1) * rows]), range(K)))
    # tagged int32, not uint32: the same bytes, and the program reads the buffer as raw words either way
    h = ttnn.from_torch(torch.from_numpy(buf.view(np.int32)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0))
    return h, ttnn.Shape([rows, words])


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


def download_state_pages(md, t, P, HPG, ranges):
    """d -> device d's homes as (P, HPG, LANES) uint32 with only the page ranges read (the rest zero). Each range is
    one slice of every core's homes on the whole mesh, so all chips cut it at once, and one read: the host reads the
    state pages and not the whole matrix, and not chip after chip."""
    import ttnn
    try:
        x = ttnn.reshape(t, ttnn.Shape([P, HPG, LANES]))
        parts = []
        for lo, hi in ranges:
            y = ttnn.slice(x, [0, lo, 0], [P, hi, LANES])
            parts.append(ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(np.uint32))
            ttnn.deallocate(y)
    except Exception as e:   # a ttnn without the mesh slice: chip by chip
        print("download_state_pages: mesh slice failed (%s), reading chip by chip" % str(e).splitlines()[0][:120], flush=True)
        return lambda d: download_home_pages(t, d, P, HPG, ranges)
    return spread_pages(parts, P, HPG, ranges)


def spread_pages(parts, P, HPG, ranges):
    """d -> (P, HPG, LANES) uint32 from one (devices * P, hi - lo, LANES) array per page range, in device order."""
    def pages(d):
        out = np.zeros((P, HPG, LANES), np.uint32)
        for (lo, hi), h in zip(ranges, parts):
            out[:, lo:hi] = h[d * P:(d + 1) * P]
        return out
    return pages


def download_home_pages(t, d, P, HPG, ranges):
    """Device d's homes as (P, HPG, LANES) uint32 with only the page ranges read (the rest zero): each range is a slice
    of every core's homes, cut on the device, so the host reads the state pages and not the whole matrix."""
    import ttnn
    out = np.zeros((P, HPG, LANES), np.uint32)
    try:
        x = ttnn.reshape(ttnn.get_device_tensors(t)[d], ttnn.Shape([P, HPG, LANES]))
        for lo, hi in ranges:
            y = ttnn.slice(x, [0, lo, 0], [P, hi, LANES])
            out[:, lo:hi] = ttnn.to_torch(y).numpy().view(np.uint32)
            ttnn.deallocate(y)
    except Exception as e:   # a ttnn without a per-device slice: read the whole homes
        print("download_home_pages: device slice failed (%s), reading all %d pages" % (str(e).splitlines()[0][:120], HPG), flush=True)
        out = download_chip(t, d).reshape(P, HPG, LANES)
    return out
