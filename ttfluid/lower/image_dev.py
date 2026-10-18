"""run_mesh.py --from-images DIR: the device run from per-chip images (lower/image.py) instead of a lowering, uploading and
downloading chip by chip so the host never holds all K chips' arrays (an occDrivAer 65M image is ~6.3 GB per chip).

    IMG, m, E, gtab, info, cells, place, SHS, NSH = setup(d, LOWER, K, P, R, C)
    tab_gw, chip_tab, L = gateway_tables(IMG, gtab, xyw, K, P)
    t = upload(md, K, DEVBLK, f, to_dev)         # f(k) -> chip k's (rows, words) uint32 array
    x = download_chip(t, d)                      # device d's shard of a mesh tensor, as numpy

m and E are stand-ins carrying exactly the attributes run_mesh's device code reads. Upload: a mesh tensor is allocated
on the device and each chip's shard is written into its device's buffer (ttnn.get_device_tensors +
copy_host_to_device_tensor), one chip's host array at a time. CFD_IMG_UPLOAD=full instead builds the whole
(K * rows, words) array and uses run_mesh's to_dev (host holds every chip: small cases, and the A/B for the chip-wise
path); CFD_IMG_VERIFY=1 reads every uploaded tensor back chip by chip and compares it with the image.
"""
import os
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


def setup(d, LOWER, K, P, R, C):
    img = Image.load(d)
    s, sh = img.s, img.shared
    flags = dict(sh["flags"])
    # the case directory may sit elsewhere on the device host: the images carry everything the run reads from it
    diff = {k: (flags.get(k), LOWER[k]) for k in LOWER if k not in ("case", "n") and flags.get(k) != LOWER[k]}
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
    E = SimpleNamespace(c=c, b=sh["b"], n=sh["n"], tab_stream=s["tab_stream"], loop_pages=s["loop_pages"],
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


def chip_arrays(img, k, P, state=None, want=None):
    """Chip k's arrays as run_mesh's device code builds them from E (arena, consts, home, blob). state: name -> global
    array (Image.state's layout) written over the arena and homes, so the run starts from it (run_mesh --state-in);
    want: the one array the caller takes, so only it is copied and written."""
    ch = img.chip(k)
    TBP = img.s["TBP"]
    out = dict(arena=pad16(np.asarray(ch["arena"]).reshape(P, -1)),
               consts=pad16(np.asarray(ch["consts"]).reshape(-1, 1024)),
               home=np.ascontiguousarray(np.asarray(ch["home"]).reshape(-1, 1024).view(np.uint32)))
    if state is not None and want in (None, "arena", "home"):   # copies: the image files are mapped read-only
        A = H = None
        if want in (None, "arena"):
            out["arena"] = out["arena"].copy()
            nt = img.s["ntiles"]
            A = out["arena"][:, :nt * 1024].view(np.float32).reshape(P, nt, 1024)
        if want in (None, "home"):
            out["home"] = out["home"].copy()
            H = out["home"].view(np.float32).reshape(P, -1, 1024)
        img.put_state(k, A, H, state)
    out["blob"] = (np.asarray(ch["blob"]).reshape(P * TBP, 1024) if img.s["tab_stream"]
                   else np.zeros((P * TBP, 1024), np.uint32))
    return out


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


def upload(md, K, DEVBLK, f, to_dev):
    """A (K * rows, words) uint32 mesh tensor whose device d holds f(DEVBLK[d]), its shards written in one copy (built one chip at a time)."""
    import ttnn
    if os.environ.get("CFD_IMG_UPLOAD") == "full":
        return to_dev(np.concatenate([f(k) for k in range(K)]))
    h, shape = host_shards(md, K, DEVBLK, f)
    t = ttnn.allocate_tensor_on_device(shape, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy_host_to_device_tensor(h, t)
    del h
    if os.environ.get("CFD_IMG_VERIFY"):
        for d in range(K):
            assert np.array_equal(download_chip(t, d), f(DEVBLK[d])), ("chip-wise upload read back differs", d)
    return t


def host_shards(md, K, DEVBLK, f):
    """upload's host side: the K shards as one multi-device host tensor and one shard's shape, ready for
    copy_host_to_device_tensor (run_mesh --cases builds the next case's homes with it while the device runs this one)."""
    import torch
    import ttnn
    # one write of K host shards: copy_host_to_device_tensor into a get_device_tensors() view writes EVERY device
    # (ttnn 0.68: all 8 chips read back the last chip's data), so the shards go in together via from_host_shards.
    # The host holds one array of every chip at a time, not every array.
    # the K host shards are built in parallel (image reads + copies release the GIL); CFD_IMG_THREADS=1 is the old serial path
    from concurrent.futures import ThreadPoolExecutor

    def shard(d):
        x = np.ascontiguousarray(f(DEVBLK[d]))
        assert x.dtype == np.uint32, (x.shape, x.dtype)
        # tagged int32, not uint32: the same bytes, and from_torch builds an int32 host tensor at ~2 GB/s against ~1 GB/s
        # for uint32 (measured with a host-to-device transfer benchmark); the program reads the buffer as raw words either way
        return ttnn.from_torch(torch.from_numpy(x.view(np.int32)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    with ThreadPoolExecutor(int(os.environ.get("CFD_IMG_THREADS", min(K, 16)))) as ex:
        hs = list(ex.map(shard, range(K)))
    assert len({tuple(h.shape) for h in hs}) == 1, [tuple(h.shape) for h in hs]
    return ttnn.from_host_shards(hs, md.shape), ttnn.Shape(list(hs[0].shape))   # the shape: one device's shard


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
