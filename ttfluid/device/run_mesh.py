"""Run a lowered machine program on a mesh of chips: the resident VM, one program per chip, the halos and the
reductions crossing chips over the fabric from inside the program (vm_brisc.cpp / vm_ncrisc.cpp, CFD_MESH).

The program is lowered once for K * P cores (P = 72 on Wormhole, 110 on Blackhole; chip blocks first, see ttfluid.resident.machine
Maps.group), so the CPU machine emulator with K * P cores is the exact reference of the K-chip run, reduction order
included. Two Blackhole chips of a p300 board have two links between them: --arch bh --gw 1 --links 0 --rlink 1.

    TT_VISIBLE_DEVICES=0,1 run_mesh.py case 400 --mesh 1,2 --case ~/cfd/resident/runs/pitzDaily
Options as run_vm.py, plus --mesh R,C (the chips' mesh; K = R * C partition blocks are placed on it so that
neighbouring blocks are neighbouring chips).
"""
import argparse, collections, itertools, json, os, pickle, re, sys, threading, time
from pathlib import Path

import numpy as np

T_START = time.time()


def stamp(what):   # host step timestamps, seconds from process start: where a launch spends its wall clock
    print("[t %7.1f s] %s" % (time.time() - T_START, what), flush=True)


HERE = Path(__file__).resolve().parent
DEV = HERE / "kernels"
# per arch: worker grid, CB bytes the auto placement may use, CBs per core and the DRAM read alignment
# (NOC_DRAM_READ_ALIGNMENT_BYTES). Blackhole on ttnn 0.68: 11x10 Tensix, 1.5 MB L1, 64 CBs. The L1 CB region is read from
# the opened device (L1_CB below).
ARCH = dict(wh=((8, 9), 1_300_000, 32, 32), bh=((11, 10), 1_370_000, 64, 64))
ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["synth", "case"])
ap.add_argument("n", type=int, nargs="?", default=5)
ap.add_argument("--mesh", default="1,2")
ap.add_argument("--case", default="cases/pitzDaily")
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--legs", type=int, default=0, help="with --from-images: chain this many launches of the n-iteration program "
                "in one process, each continuing from the last (DRAM homes stay on the device, the arena's state registers "
                "come back from the last run's dump): a whole case per process, the same answer as a --state-in chain")
ap.add_argument("--cases", type=int, default=0, help="with --from-images: this many whole cases back to back in one process "
                "(sweep mode), each from the images' start (or --state-in), each with its own state npz and result row; the mesh "
                "opens once and the next case's DRAM homes are built on the host while the device runs this one")
ap.add_argument("--overlap", type=int, default=1, help="--cases: 1 = case k's outputs are copied on the device, case k+1 is "
                "launched, and case k is read and saved over a second command queue while k+1 runs (the next homes are "
                "written there too); 0 = the serial path (read, save, write, launch)")
ap.add_argument("--out", default="results_mesh.jsonl")
ap.add_argument("--noemu", action="store_true")
ap.add_argument("--write", default="", help="as run_vm.py: write the final fields (OpenFOAM format) into this case dir copy")
ap.add_argument("--layout", default="lattice", choices=["lattice", "rcb"])
ap.add_argument("--tiles", type=int, default=None)
ap.add_argument("--solve-census", help="CPU: after lowering, write tools/solve_census.py's per-core pages of every Krylov loop pass "
                "and outer phase to this JSON (cores from --sc-cores) and exit")
ap.add_argument("--sc-cores", default="0", help="--solve-census: comma list of global cores")
ap.add_argument("--cores", type=int, default=None, help="CPU only (--plan / --emuonly): lower for this many cores per chip instead of "
                "the --arch grid's, so a small case reaches a big case's cells per core (and its T)")
ap.add_argument("--l1", default=None)
ap.add_argument("--ring", type=int, default=4, help="as run_vm.py: pages in the DRAM reader and writer rings")
ap.add_argument("--l1-bytes", type=int, default=None, help="CB bytes per core the auto placement may use (default: per --arch)")
ap.add_argument("--arch", default="wh", choices=list(ARCH), help="wh: Wormhole n150/Galaxy chips; bh: Blackhole p150a/p300c chips")
ap.add_argument("--hist", type=int, default=4096)
ap.add_argument("--limit", type=int, default=1_500_000_000)
ap.add_argument("--bnoc", type=int, default=1, help="NoC of the B agent on BRISC (N gets the other)")
ap.add_argument("--nsh", type=int, default=2, help="gateway shadow regions, shared by the exchanges round-robin")
ap.add_argument("--gw", type=int, default=1, help="halo gateways per chip pair: the receivers split among them by words, each gateway on its own link")
ap.add_argument("--links", default="0,1,2", help="ethernet links of the halo gateways, gateway k of a pair on the k-th (the reduce uses --rlink)")
ap.add_argument("--rlink", type=int, default=3, help="ethernet link of the cross-chip reductions (WH 3; a p300 board has links 0 and 1 only)")
ap.add_argument("--red2", type=int, default=-1, help="two-stage reductions: -1 auto (K >= 8 or K * P > 1024), 0 off, 1 on")
ap.add_argument("--emuonly", action="store_true", help="CPU only: lower and emulate, print the last row")
ap.add_argument("--devsim", action="store_true", help="with --emuonly: run the encoded streams through the protocol simulator "
                "(ttfluid/resident/devsim.py, token depth 64) to catch a deadlock or a token count mismatch before the device does")
ap.add_argument("--emu-check", action="store_true", help="memu checks every read against the register its tile holds (and wplace 2 overlaps)")
ap.add_argument("--ref", help="emulator reference file: --emuonly writes it, a device run reads it instead of emulating, so the "
                "emulator (minutes at 576+ cores) runs outside the chip locks and their bound")
ap.add_argument("--fuse", action="store_true", help="segment tile-loop fusion (Machine(fuse=True))")
ap.add_argument("--redm", type=int, default=1, help="1: pair adjacent independent global sums into one reduction (same values, fewer all-gathers); 0: one per sum")
ap.add_argument("--owner", help="npz with key parts: the global core of every cell (drivaer part2d), in place of rcb")
ap.add_argument("--strips", action="store_true", help="chip blocks as strips along the longest axis on any R x C mesh (snake placement)")
ap.add_argument("--block2d", action="store_true", help="R x C is the chips' own physical shape (a .110 tray of 8 is 2,4); without it a 2D mesh is refused, because a view the chips do not have (4,2 on a 2x4 tray) routes over hops that are not links and wedged all eight (.110 0-7, 2026-10-10)")
ap.add_argument("--page", type=int, default=0, help="as run_vm.py: compute code paged from DRAM through this many 1-KB pages")
ap.add_argument("--nwin", type=int, default=0, help="as run_vm.py: N code window in L1 (words, multiple of 256)")
ap.add_argument("--l1-check", action="store_true", help="open the mesh, size every CB, build the program and enqueue it with no work (limit 0, no upload): exit 4 if it is over the kernel config buffer")
ap.add_argument("--gcache", type=int, default=-1, help="as run_vm.py: N gather tile cache slots of 4 KB; -1 (default): T + 1, a whole vector, "
               "so a DRAM-homed gather source is fetched once per part whatever T is; fewer if the L1 is short")
ap.add_argument("--icache", type=int, default=1, help="as run_vm.py: compute skips a repeated op _init (CFD_ICACHE)")
ap.add_argument("--sched", type=int, default=3, help="as run_vm.py: 1 greedy reorder of each fused run's blocks so values die early, 2 greedy or depth-first, 3 (default) the same at min-store cut points (Machine(sched=K))")
ap.add_argument("--dedup", type=int, default=0, help="as run_vm.py: a fused run streams each DRAM page once (Machine(dedup=True))")
ap.add_argument("--tsort", type=int, default=None, help="as run_vm.py: tabstream residue entries sorted by source tile (CFD_TSORT). "
                "Default on with --tabstream; with --from-images as the images were lowered")
ap.add_argument("--bpatch", type=int, default=None, help="as run_vm.py: -1 / unset auto (4 when --tabstream and >= --bpatch-min local residue entries per patched map tile), K in 1..7 BRISC scatters (8 - K) / 8 of a patched page's L1 residue (CFD_BPATCH; mailbox in the last 256 B of cb_sync)")
ap.add_argument("--bpatch-min", type=float, default=64.0)
ap.add_argument("--bgts", type=int, default=0, help="K in 1..7: BRISC scatters the upper (8 - K) / 8 of every L1-source and halo section of a "
                "--tabstream GATHER tile, NCRISC the rest (CFD_BGTS, the --bpatch mailbox); 0 off. Data movement only: bit-exact")
ap.add_argument("--tgts", type=int, default=0, help="J in 1..7: compute's UNPACK copies the top J / 8 of every L1-source and halo section of a "
                "--tabstream GATHER tile (of what --bgts leaves NCRISC) from inside its waits (CFD_TGTS, a second mailbox); 0 off. Bit-exact")
ap.add_argument("--amg", action="store_true", help="p PCG preconditioned by smoothed aggregation (ttfluid.resident.sa), aggregates on the K x 72 global cores; "
                "the coarse levels are chip-local (sa.setup chip_cores): only the fine level exchanges between chips")
ap.add_argument("--amg-kind", default="sa", choices=["sa"])
ap.add_argument("--amg-stack", action="store_true", help="as run_vm.py: stacked coarse SpMV")
ap.add_argument("--amg-vec", action="store_true", help="as run_vm.py: coarse values are vector registers (--l1 auto may home them in DRAM)")
ap.add_argument("--sliced", action="store_true", help="sorted rcb lanes; report what sliced ELL slots would save (machine.slot_extents)")
ap.add_argument("--lsched", action="store_true", help="reorder straight-line IR runs for fewer live vectors (machine._lsched)")
ap.add_argument("--ptiles", action="store_true", help="wide patch spaces take their own tile count (default: the cells' T)")
ap.add_argument("--amg-smoothed", type=int, default=99)
ap.add_argument("--amg-lag", type=int, default=1)
ap.add_argument("--amg-lag-warm", type=int, default=0)
ap.add_argument("--amg-depth", type=int, default=None, help="at most this many coarse levels")
ap.add_argument("--amg-min-core", type=int, default=0, help="no coarse level with fewer than this many aggregates per core (r13 2x4: 10 picks depth 3, the fastest)")
ap.add_argument("--vpre", type=int, default=3, help="sa V-cycle: Chebyshev degree of the pre smoothing on every level above the coarsest")
ap.add_argument("--vpost", type=int, default=3, help="sa V-cycle: Chebyshev degree of the post smoothing (0: none, an unsymmetric preconditioner)")
ap.add_argument("--vcoarse", type=int, default=8, help="sa V-cycle: Chebyshev degree on the coarsest level (sa.COARSE_DEG)")
ap.add_argument("--vgamma", default="", help="sa cycle: comma list of levels taking two coarse corrections (a W at each), e.g. 1")
ap.add_argument("--amg-global", action="store_true", help="CPU reference only (--emuonly / --plan): the multigrid over all K x 72 cores, coarse levels across chips")
ap.add_argument("--hoist", type=int, default=0, help="Machine(hoist=True): a gather, halo or reduction independent of a fused run moves around it, so a cross-chip wait can sit behind interior work")
ap.add_argument("--wide", action="store_true", help="22-bit gather entries even at T <= 64 (forced on above: a test of the T > 64 format)")
ap.add_argument("--tscheck", action="store_true", help="with --tabstream: replay the streamed table walk on the CPU (tabstream.check)")
ap.add_argument("--tabstream", action="store_true", help="as run_vm.py: residue tables streamed from DRAM (ttfluid/resident/tabstream.py), per chip")
ap.add_argument("--stop", type=int, default=0, help="bisect a hang: every core aborts on purpose (reason 90/91/92) after 1 the start barrier, 2 the first chip all-gather, 3 the first halo exchange")
for k, h in [("jointred", "krylov.JOINT + Machine(joint_red=True)"), ("cg1", "krylov.CG1"), ("fold1", "Machine(fold1=True)"),
             ("fusepart", "Machine(fusepart=True)"), ("gsplit", "Machine(gsplit=True)"), ("wplace", "Machine(wplace=True)"),
             ("bcast", "Machine(bcast_run=True)"), ("bgather", "Machine(bgather=True)"), ("bscal", "Machine(bscal=True)"),
             ("bjoint", "krylov.BJOINT (pair with --jointred 1)"), ("cse", "Machine(cse_ir=True): value numbering over the traced IR, bit-exact"), ("narrow", "bf16 in-place DRAM constants (Machine(narrow=True), C_LDRH)")]:
    ap.add_argument("--" + k, type=int, default=0, help="as run_vm.py: " + h)
ap.add_argument("--gtile", type=int, default=0, help="1 (with --tabstream): a gather whose next sync hands it to a tile loop that reads it "
                "plainly posts one token per finished tile and compute waits once per pass, so the loop starts on tile 0 "
                "while NCRISC fills the rest (encode.py gtile). Encoded into the streams: needs a lowering. Same values")
ap.add_argument("--hlate", type=int, default=0, help="1: fewer finest halos live at once (smaller halo buffer, more L1 for --l1): x[idx] gathers "
                "component-major (trace.FIELD_MAJOR) and each halo is sent right before its first gather (Machine(hlate=True))")
ap.add_argument("--hre", type=int, default=0, help="G > 0: a halo whose last gather is more than G MIR back is sent again "
                "instead of held in its slot (fewer finest halos live at once, more halo sends; Machine(hre=G))")
ap.add_argument("--forces", default=None, metavar="FUNC", help="Cd/Cl of each outer iteration's incoming state in the history (result keys Cd_in, Cl_in), from the controlDict function FUNC (ttfluid.solver.forces.coeff_hook)")
ap.add_argument("--phase", action="store_true", help="per-SIMPLE-phase device cycles (program.PHASES markers, CFD_PHASE): result key phase")
ap.add_argument("--cap", type=int, default=32, help="as run_vm.py: scratch cap, live one-tile values per fused run")
ap.add_argument("--pseg", type=int, default=0, help="as run_vm.py: a split gather joins its fused run as one reader-patched page (Machine(pseg)); 3, 4 need --tabstream (mesh has no trid). Default 0 keeps the built images unchanged")
ap.add_argument("--os", type=int, default=0, help="1: BRISC built for size (#pragma GCC optimize Os), 2: NCRISC too, for a program over the kernel config buffer")
ap.add_argument("--prof", action="store_true", help="census build: per-op cycles of every interpreter (CFD_PROF), means over all K x 72 cores")
ap.add_argument("--plan", action="store_true", help="CPU only: lower, place the chips, print the plan")
ap.add_argument("--gruns", help="CPU only: lower and encode (--tabstream), write the gather run census (lower/gruns.py) of the "
                "cores of --gruns-chips to this json and exit")
ap.add_argument("--gruns-chips", default="0,8,16,31")
ap.add_argument("--save", help="CPU only: lower and encode, pickle the result to this file and exit, so a large case lowers "
                "off-chip and the device run (--load) holds its chips only for the run")
ap.add_argument("--gdump", help="CPU: dump every gather op and the local gather entries of --gdump-cores (tools/gcensus.py), exit before encoding")
ap.add_argument("--gdump-cores", default=None, help="comma list of global cores (default: every 11th core of chip 0)")
ap.add_argument("--mcache", help="CPU: pickle the lowered Machine here, or read it if the file exists (encode-only rerun); *.zst: zstd-compressed")
ap.add_argument("--load", help="read a --save file instead of lowering; the lowering flags must be the ones it was saved with")
ap.add_argument("--from-images", help="device run from the per-chip images in this directory (lower/image_dev.py), uploaded chip by "
                "chip; the lowering flags must be the ones they were written with")
ap.add_argument("--state-in", help="with --from-images: start from this state (an earlier run's _state.npz, or lower/of_state.py "
                "from an OpenFOAM time directory) instead of the images' initial fields: a restart")
ap.add_argument("--images", help="CPU only: lower, write per-chip device images to this directory (lower/image.py) and exit; "
                "their sha256 per chip goes to digests.json there")
ap.add_argument("--min-free-gb", type=int, default=200, help="refuse an --images/--save/--mcache write that would leave its disk "
                "under this many GB free (checked before lowering, and again with the image's exact size before writing it)")
a = ap.parse_args()
if a.tsort is None and not a.from_images:
    a.tsort = int(a.tabstream)
for out in (a.images, a.save, a.mcache and not Path(a.mcache).exists() and a.mcache):
    if out:   # fail before a lowering that can take an hour, not after it
        from ttfluid.lower.image import check_free
        check_free(out, 0, a.min_free_gb)
R, C = map(int, a.mesh.split(","))
assert min(R, C) == 1 or a.block2d or a.plan or a.emuonly, "2D mesh %dx%d needs --block2d (chips must be a physical block)" % (R, C)
GRID, l1_default, NCB, DA = ARCH[a.arch]
a.l1_bytes = a.l1_bytes or l1_default
K, P = R * C, GRID[0] * GRID[1]
assert not a.cores or a.plan or a.emuonly or a.solve_census, "--cores: a CPU lowering only (the device grid is --arch's)"
P = a.cores or P
PP = -(-P // 4) * 4   # one chip's partials padded to 16 B (vm_brisc.cpp PP): P = 110 on BH, 72 on WH (unchanged)
import ttfluid.resident.encode  # noqa: E402
ttfluid.resident.encode.DA = DA

from ttfluid.solver.simple import SimpleFoam  # noqa: E402
from ttfluid.resident.machine import Machine  # noqa: E402
if a.owner:
    import ttfluid.resident.machine
    _parts = np.load(a.owner)["parts"].astype(np.int64)
    ttfluid.resident.machine.rcb = lambda xyz, P, chips=None: _parts
from ttfluid.resident.memu import MachineEmu  # noqa: E402
from ttfluid.resident.encode import Encoded  # noqa: E402
from ttfluid.resident.program import build  # noqa: E402
import ttfluid.resident.krylov  # noqa: E402

ttfluid.resident.krylov.JOINT = bool(a.jointred)
VSHAPE = None                     # sa.Ops cycle shape; None = today's 3/3/8 V (the default build is unchanged)
if (a.vpre, a.vpost, a.vcoarse, a.vgamma) != (3, 3, 8, ""):
    VSHAPE = dict(pre=a.vpre, post=a.vpost, coarse_deg=a.vcoarse, gamma={int(v): 2 for v in a.vgamma.split(",") if v})
import ttfluid.resident.trace  # noqa: E402
ttfluid.resident.trace.FIELD_MAJOR = bool(a.hlate)
ttfluid.resident.krylov.CG1 = bool(a.cg1)
ttfluid.resident.krylov.BJOINT = bool(a.bjoint)
stamp("python imports done")

t0 = time.time()
# the flags that shape the lowering and the encoding: a --load must carry the ones its --save had
LOWER = {k: getattr(a, k) for k in ("what", "n", "case", "mesh", "layout", "tiles", "l1", "red2", "strips", "fuse", "redm", "sched",
                                    "hoist", "jointred", "fold1", "fusepart", "gsplit", "wplace", "bcast", "cg1", "page", "tabstream",
                                    "bgather", "bscal", "bjoint", "cap")}
if a.phase:
    LOWER["phase"] = True
if a.tsort:   # images and --save files from before the flag have unsorted tables, so 0 stays out of LOWER
    LOWER["tsort"] = 1
# flags added since: in LOWER only when not at their default, so an image or --save without them still matches
LOWER.update({k: getattr(a, k) for k in ("cse", "dedup", "narrow", "amg", "amg_kind", "amg_stack", "amg_vec", "ptiles", "lsched", "sliced", "amg_smoothed", "amg_lag",
                                         "amg_lag_warm", "amg_depth", "amg_min_core", "amg_global", "hlate", "hre", "gtile", "vpre", "vpost", "vcoarse", "vgamma", "forces", "pseg") if getattr(a, k) != ap.get_default(k)})
assert a.pseg < 3 or a.tabstream, "--pseg 3, 4 patch many pages per stream: the reader does that with --tabstream only"
assert not a.amg_global or a.emuonly or a.plan, "--amg-global: a CPU reference (the gateway carries the cell plan only)"
IMG = None

def mcache_open(mode):   # a --mcache path ending in .zst is read and written as a zstd stream (Small k4: 121 GB raw)
    if not a.mcache.endswith(".zst"):
        return open(a.mcache, mode)
    import io
    import zstandard
    f = open(a.mcache, mode)
    if mode == "rb":
        return io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True, closefd=True), 1 << 24)
    return zstandard.ZstdCompressor(level=3, threads=6).stream_writer(f, closefd=True)

if a.from_images:   # the device run from per-chip images (lower/image_dev.py): no lowering, chip-by-chip upload
    from ttfluid.lower import image_dev  # noqa: E402
    assert a.noemu or a.ref, "--from-images: no lowering to emulate (--noemu, or --ref a saved reference)"
    IMG, m, E, gtab, info, cells, place, SHS, NSH = image_dev.setup(a.from_images, LOWER, K, P, R, C)
    ttfluid.resident.encode.WIDE = a.wide or m.T > 64   # the images were lowered so (below)
    if IMG.shared["flags"].get("phase") and not a.phase:   # images lowered with --phase carry the markers: run with CFD_PHASE
        a.phase = True
        print("images lowered with --phase: CFD_PHASE on", flush=True)
    if a.tsort is None:   # CFD_TSORT reads four entries per cached tile: only right on tables lowered sorted
        a.tsort = int(IMG.shared["flags"].get("tsort", 0))
    mp = m.maps
    NPL = len(getattr(mp, "plans", [None]))
    chip = lambda g: g // P
    LINKS = [int(x) for x in a.links.split(",")]
    assert a.rlink not in LINKS and all(v["lane"] < len(LINKS) for v in gtab.values()), \
        ("--links: one per gateway lane the images were lowered with (--gw), the reduce on its own --rlink", LINKS, a.rlink)
    pid_at = {cells[place[k]]: k for k in range(K)}
    print("images %s: %s" % (a.from_images, json.dumps(info)), flush=True)
    stamp("images set up")
    assert not a.tabstream or m.ndram, "--tabstream streams pages: it needs DRAM-homed vectors (--l1)"
    assert not a.page or E.loop_pages < a.page, f"a tile loop spans {E.loop_pages} code pages: --page must exceed it"
    NMAX = E.n_maxop()
    if a.nwin and a.nwin < NMAX + 255:
        a.nwin = -(-(NMAX + 255) // 256) * 256
        print("--nwin raised to %d words (longest N op %d)" % (a.nwin, NMAX), flush=True)
else:
    if a.load:
        with open(a.load, "rb") as f:
            saved, prog, m, E = pickle.load(f)
        saved = {**{k: ap.get_default(k) for k in LOWER}, **saved}   # a key added since the --save had its default
        assert saved == LOWER, ("--load: lowering flags differ from the --save", {k: (saved[k], LOWER[k]) for k in LOWER if saved[k] != LOWER[k]})
        print("loaded %s in %.1f s" % (a.load, time.time() - t0), flush=True)
    elif a.mcache and os.path.exists(a.mcache):
        with mcache_open("rb") as f:
            saved, prog, m = pickle.load(f)
        saved, low = ({k: v for k, v in d.items() if k != "tsort"} for d in (saved, LOWER))   # tsort is in the encoding, not the Machine
        assert saved == low, ("--mcache: lowering flags differ", {k: (saved.get(k), low[k]) for k in low if saved.get(k) != low[k]})
        print("machine from %s in %.1f s" % (a.mcache, time.time() - t0), flush=True)
    else:
        s = SimpleFoam(a.case, backend="ell", dtype=np.float32, log=lambda *x: None,
                       ell_order="dir" if a.layout == "lattice" else "face")
        core = None
        if a.layout == "rcb" and K > 1:   # chip placement first (seconds): an unplaceable partition used to surface at plan(), an hour in
            import ttfluid.resident.machine
            from ttfluid.resident import chipplan
            core = ttfluid.resident.machine.rcb(s.mesh.C, K * P, chips=K if a.strips or min(R, C) == 1 else None)
            nI = s.mesh.nInternalFaces
            chipplan.place_blocks(chipplan.chip_adjacency(core // P, s.mesh.owner[:nI], s.mesh.neighbour, K), K, R, C)
            print("placement ok: %d chip blocks one-hop on %dx%d" % (K, R, C), flush=True)
        if a.forces:
            from ttfluid.solver.forces import coeff_hook
            s.force_hook = coeff_hook(s, a.forces)
        if a.what == "synth":
            from ttfluid.resident.synth import synth
            prog = synth(s, a.n)
        elif a.amg:
            import ttfluid.resident.machine
            assert a.layout == "rcb", "--amg: rcb (the owner below is Machine's rcb)"
            owner = core if core is not None else ttfluid.resident.machine.rcb(s.mesh.C, K * P, chips=K if a.strips or min(R, C) == 1 else None)
            prog = build(s, a.n, phase=a.phase, amg_owner=owner, amg_kind=a.amg_kind, amg_smoothed=a.amg_smoothed, amg_stack=a.amg_stack,
                         amg_lag=a.amg_lag, amg_lag_warm=a.amg_lag_warm, amg_chip_cores=None if a.amg_global else P,
                         amg_depth=a.amg_depth, amg_min_core=a.amg_min_core,
                         amg_vshape=VSHAPE)
        else:
            prog = build(s, a.n, phase=a.phase)
        m = Machine(prog, s.mesh, K * P, layout=a.layout, tiles=a.tiles, chip_cores=P, red2=None if a.red2 < 0 else bool(a.red2), chip_line=a.strips or min(R, C) == 1,
                    l1_vec=None if a.l1 in (None, "auto") else int(a.l1), fuse=a.fuse, redm=bool(a.redm), sched=a.sched, hoist=bool(a.hoist),
                    joint_red=bool(a.jointred), fold1=bool(a.fold1), fusepart=bool(a.fusepart), gsplit=bool(a.gsplit), wplace=a.wplace,
                    bcast_run=bool(a.bcast), bgather=bool(a.bgather), bscal=bool(a.bscal), scratch_cap=a.cap, narrow=a.narrow,
                    amg_vec=a.amg_vec, ptiles=a.ptiles, lsched=a.lsched, sliced=a.sliced, dedup=bool(a.dedup), cse_ir=bool(a.cse), hlate=bool(a.hlate), hre=a.hre, pseg=a.pseg)
        if a.gdump:
            from ttfluid.tools import gcensus
            gcensus.dump(m, prog, a.gdump, [int(x) for x in a.gdump_cores.split(",")] if a.gdump_cores else list(range(0, P, 11)))
            sys.exit(0)
        if a.mcache:   # the Machine before encoding: an encode-only rerun skips the lowering (Small k1: 17 min, 98 GB)
            import cloudpickle
            with mcache_open("wb") as f:
                cloudpickle.dump((LOWER, prog, m), f, protocol=5)
            print("machine cached to %s at %.1f s" % (a.mcache, time.time() - t0), flush=True)
    if a.solve_census:
        from ttfluid.tools import solve_census
        Path(a.solve_census).write_text(json.dumps(solve_census.report(prog, m, [int(c) for c in a.sc_cores.split(",")], a.gcache if a.gcache >= 0 else m.T + 1), indent=1))
        sys.exit(print("solve census -> %s at %.1f s" % (a.solve_census, time.time() - t0), flush=True))
    mp = m.maps
    chip = lambda g: g // P
    NPL = len(getattr(mp, "plans", [None]))
    for q, pq in enumerate(getattr(mp, "plans", [])[1:], 1):    # the gateway carries plan 0 (cells) only
        xs = [(j, i) for (j, i) in pq["send"] if chip(j) != chip(i)]
        assert a.amg_global or not xs, ("halo plan %d (%s) crosses chips" % (q, pq["space"]), xs[:4])


    # ---- the chips' halo plan: gateways, shadows, placement (ttfluid.resident.chipplan, shared with the per-chip lowering)
    from ttfluid.resident import chipplan  # noqa: E402
    cross = {(j, i): (mp.recv_off[(j, i)], len(mp.send[(j, i)])) for (j, i) in mp.send if chip(j) != chip(i)}
    pl = chipplan.plan(cross, K, P, R, C, gw=a.gw, nsh=a.nsh, nhalo=m.nhalo, hmax=mp.hmax,
                       runs={key: (mp.recv_off[key], len(mp.send[key])) for key in mp.send})
    pairs, nbrs, gw, lane, place, cells = pl["pairs"], pl["nbrs"], pl["gw"], pl["lane"], pl["place"], pl["cells"]
    LINKS = [int(x) for x in a.links.split(",")]
    assert a.gw <= len(LINKS) and a.rlink not in LINKS, ("--gw needs one link per gateway, the reduce has its own", a.gw, LINKS, a.rlink)
    pid_at = {cells[place[k]]: k for k in range(K)}

    assert a.page & (a.page - 1) == 0, "--page: a power of 2"
    SHS, NSH = pl["SHS"], pl["NSH"]            # words per shadow region, shadow regions (ttfluid.resident.chipplan)


    def cb_bytes(E):
        mm = E.m
        pad = lambda n: -(-n // 16) * 64
        fixed = 4096 + 16384 + 8 * 64 * 32 + 32 + 4096 + -(-mm.hwords * 4 // 64) * 64
        rings = (a.ring * (4096 + 32) + a.ring * 4096 + 4096 + 512 + 512) if mm.ndram else 256
        mesh = 8192 + 512 + 2 * K * (16 if mm.red2 else 12 * PP) + NSH * SHS * 4 + 512
        code = (a.page * 1024 if a.page else pad(len(E.c))) + pad(len(E.b)) + (a.nwin * 4 if a.nwin else pad(len(E.n)))
        ts = 2 * (4096 + 64 * 32) if E.tab_stream else 0
        ts += 4096 if E.tab_stream and a.bpatch != 0 else 0      # cb_sync may grow a page for the B-patch mailbox
        return mm.ntiles * 4096 + code + pad(E.tab.shape[1] + 64) + fixed + rings + mesh + max(a.gcache, 6 if a.gcache < 0 else 0) * 4096 + ts + (0 if mm.ndram else 2048 * min(a.narrow, 2))


    ttfluid.resident.encode.WIDE = a.wide or m.T > 64   # over 65,536 lanes a core: 22-bit gather entries (ttfluid/resident/encode.py)
    assert a.tabstream or not ttfluid.resident.encode.WIDE, "T=%d > 64 needs --tabstream" % m.T
    assert not (ttfluid.resident.encode.WIDE and (a.tgts or a.gtile)), "--tgts / --gtile hand over 16-bit gather entries (vm_compute tsvc): not with wide ones"
    if not a.load:
        E = Encoded(m, page=bool(a.page), pdev=P, tab_stream=a.tabstream, tab_sort=bool(a.tsort), phase=getattr(prog, "phase_at", None), gtile=bool(a.gtile))
        if a.tscheck:   # as run_vm.py: NCRISC's streamed walk replayed against the cursor tables, every core
            from ttfluid.resident.tabstream import check
            check(E, E.tab_stream, E.tab_cursor)
            print("tabstream check: streamed walk == cursor walk on every core, map and tile (wide %d)" % ttfluid.resident.encode.WIDE, flush=True)
        if a.l1 == "auto" and cb_bytes(E) > a.l1_bytes:   # as run_vm.py: as many vector slots in L1 as fit
            k = max(0, int((a.l1_bytes - (cb_bytes(E) - m.ntiles * 4096)) // 4096 - m.nsmall) // m.T)
            while True:
                m.place(k, print)
                E = Encoded(m, page=bool(a.page), pdev=P, tab_stream=a.tabstream, tab_sort=bool(a.tsort), phase=getattr(prog, "phase_at", None), gtile=bool(a.gtile))
                print("l1 auto: %d vector slots in L1, CB bytes %d" % (k, cb_bytes(E)), flush=True)
                if cb_bytes(E) <= a.l1_bytes or k == 0:
                    break
                k -= 1
        if a.gtile:
            print("gtile: %d gathers hand over per tile" % E.gtile_n, flush=True)
    if a.gruns:
        from ttfluid.lower import gruns
        assert E.tab_stream, "--gruns reads the tabstream tables"
        rows = [r for k in map(int, a.gruns_chips.split(",")) for r in range(k * P, (k + 1) * P)]
        out = gruns.census(m, prog, ((E.tab[r], E.tab_blob[r]) for r in rows), log=lambda x: print(x, flush=True))
        Path(a.gruns).write_text(json.dumps(out, indent=1))
        sys.exit(0)
    if a.save:
        with open(a.save, "wb") as f:
            import cloudpickle                       # the lowered machine holds local lambdas (ttfluid.resident.machine.coalesce)
            cloudpickle.dump((LOWER, prog, m, E), f, protocol=5)
        print("saved %s, lowered + encoded in %.1f s" % (a.save, time.time() - t0), flush=True)
        sys.exit(0)
    assert not a.tabstream or m.ndram, "--tabstream streams pages: it needs DRAM-homed vectors (--l1)"
    assert not a.page or E.loop_pages < a.page, f"a tile loop spans {E.loop_pages} code pages: --page must exceed it"
    NMAX = E.n_maxop()
    if a.nwin and a.nwin < NMAX + 255:
        a.nwin = -(-(NMAX + 255) // 256) * 256
        print("--nwin raised to %d words (longest N op %d)" % (a.nwin, NMAX), flush=True)
    tab = [list(row) for row in E.tab]
    chipplan.rewrite(tab, E.tab_halo, range(K * P), pl, 1 if a.amg_global else NPL)   # halo sections: local ids, remote via gateways
    gtab = pl["gtab"]
    info = dict(K=K, mesh=[R, C], place={k: cells[place[k]] for k in range(K)}, T=m.T, hmax=mp.hmax, nhalo=m.nhalo, shs=SHS, nsh=NSH,
                cross_pairs={f"{x}->{y}": len(v) for (x, y), v in pairs.items()},
                gateways={f"{x}->{y}" + (f".{k}" if a.gw > 1 else ""): g for (x, y, k), g in gw.items()}, shadow_kb=NSH * SHS * 4 / 1024,
                gw_load={g: (v["gP"], v["gR"], len(v["rec"])) for g, v in gtab.items()},
                cells_per_chip=[int(np.sum(m.lay.owner[prog.cells.name] // P == k)) for k in range(K)])
    print("plan", json.dumps(info), flush=True)
    if a.images:
        from ttfluid.lower.image import Image
        t1 = time.time()
        img = Image.from_lowering(m, E, tab, gtab, info, K, P, LOWER)
        img.write(a.images, a.min_free_gb)
        dig = {k: img.digest(k) for k in range(K)}
        Path(a.images, "digests.json").write_text(json.dumps(dig, indent=1))
        print("images %s written in %.1f s, digests %s" % (a.images, time.time() - t1, json.dumps(dig)), flush=True)
        sys.exit(0)


    if cb_bytes(E) > a.l1_bytes:
        print("warning: CB bytes %d over --l1-bytes %d" % (cb_bytes(E), a.l1_bytes), flush=True)
    print("lowered + encoded in %.1f s" % (time.time() - t0), E.sizes(), flush=True)
    if a.plan:
        sys.exit(print("cb_bytes estimate %d" % cb_bytes(E)))

ref = None
if a.ref and not a.emuonly:
    with open(a.ref, "rb") as f:
        ref = pickle.load(f)
    print("emulator reference from", a.ref, flush=True)
elif not a.noemu:
    t0 = time.time()
    me = MachineEmu(m, np.float32)
    me.check = a.emu_check
    me.run()
    ref = (me.history, me.state())
    print("emulated (%d cores) in %.1f s, %d rows" % (K * P, time.time() - t0, len(me.history)), flush=True)
    print("executed tile ops %d, load tiles %d (per core)" % (me.optiles["op"], me.optiles["ld"]), flush=True)
    for ph_, c_ in me.phtiles.items():   # per SIMPLE phase over the whole run (build(phase=True) markers)
        print("emu phase %-12s %s" % (ph_, " ".join("%s %d" % kv for kv in sorted(c_.items()))), flush=True)
    if a.emuonly:
        for k_, r_ in enumerate(me.history[:5] + me.history[5:][-1:]):
            print("emu row", k_ if k_ < 5 else len(me.history) - 1, " ".join(f"{n} {v:.6g}" for n, v in sorted(r_.items())), flush=True)
        print("red2", m.red2, "last", json.dumps(me.history[-1]), flush=True)
        if a.devsim:
            from ttfluid.resident.devsim import DevSim
            t0 = time.time()
            r_ = DevSim(E, me.branches, depth=64, ring=a.ring, code_ring=a.page or 16).run(max_steps=10 ** 11)
            print("devsim ok in %.1f s" % (time.time() - t0), r_, flush=True)
        if a.ref:
            with open(a.ref, "wb") as f:
                pickle.dump(ref, f)
        sys.exit()

stamp("host tables built")
import torch  # noqa: E402
import ttnn  # noqa: E402

umd = [int(x) for x in os.environ["TT_VISIBLE_DEVICES"].split(",")]
assert len(umd) == K
FAB = K > 1   # one chip: no fabric at all (a single BH card never starts its Ethernet cores)
if FAB:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
stamp("python up, opening the mesh")
OVL = bool(a.cases > 1 and a.overlap)
md = ttnn.open_mesh_device(ttnn.MeshShape(R, C), num_command_queues=2 if OVL else 1)
stamp("mesh + fabric up")
# the CB region runs from the allocator's L1 base to the top of L1. Metal sets that base when it opens the device, above a
# kernel config buffer whose size depends on the board and its fabric, so it is the device's L1 + L1 small banks, not a
# per-arch constant
L1_CB = sum(ttnn._ttnn.device.GetMemoryView(md, b).total_bytes_per_bank for b in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL))
gsz = md.compute_with_storage_grid_size()
assert (gsz.x, gsz.y) == GRID, ((gsz.x, gsz.y), GRID, "--arch names the wrong grid")
cores = [(i % gsz.x, i // gsz.x) for i in range(P)]
grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gsz.x - 1, gsz.y - 1))])
phys = lambda x, y: md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
xy = [phys(*c) for c in cores]
xyw = [int(q.x) | int(q.y) << 16 for q in xy]
p0, p1 = xy[0], phys(gsz.x - 1, gsz.y - 1)
node = {rc: md.get_fabric_node_id(ttnn.MeshCoordinate(*rc)) for rc in cells} if FAB else {}

# the gateway tables now that core coordinates are known
if IMG is not None:
    tab_gw, chip_tab, L = image_dev.gateway_tables(IMG, gtab, xyw, K, P)
else:
    L0 = max(len(t) for t in tab)
    tab_gw = [0] * (K * P)
    for g, v in gtab.items():
        w = [v["gR"], v["gP"], xyw[v["peer"] % P], len(v["rec"])]
        for (i, q, lo, n, cnt) in v["rec"]:
            w += [xyw[i % P], q, lo, n, cnt]
        w += [len(v["mem"])]
        for (j, k) in v["mem"]:
            w += [j % P, k]
        tab_gw[g] = L0
        tab[g] = tab[g] + [0] * (L0 - len(tab[g])) + w
    L = max(len(t) for t in tab)
    tabarr = np.zeros((K * P, L), np.uint32)
    for g, t in enumerate(tab):
        tabarr[g, :len(t)] = t


def pad16(arr):
    arr = np.ascontiguousarray(np.asarray(arr).view(np.uint32))
    if arr.ndim == 1:
        arr = arr[None]
    w = -(-arr.shape[1] // 16) * 16
    out = np.zeros((arr.shape[0], w), np.uint32)
    out[:, :arr.shape[1]] = arr
    return out


shard = ttnn.ShardTensorToMesh(md, dim=0)


# ShardTensorToMesh hands shard d to the mesh device at row-major position d, which holds block pid_at[cells[d]]: with a
# snake or permuted placement (2D) the blocks are not in that order, so every per-chip array is permuted on the way in
# and back on the way out (a 2x4 --strips run gave row 1's chips other blocks' tables and aborted on every core)
DEVBLK = [pid_at[cells[d]] for d in range(K)]


def to_dev(arr):   # (K * rows, words) in block order: block k's rows land on the device that runs block k
    arr = np.ascontiguousarray(arr)
    arr = arr.reshape(K, -1, *arr.shape[1:])[DEVBLK].reshape(arr.shape)
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr).view(np.int32)), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=shard)


def from_dev(t, dt):   # back to block order
    x = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(dt)
    return x.reshape(K, -1, *x.shape[1:])[list(place)].reshape(x.shape)


def per_chip(f):
    return np.concatenate([f(k) for k in range(K)])


nv = m.ntiles
HPG = max(m.hpages, 1)
hmax = max(mp.hmax, 1)
H = a.hist
code = {k: pad16(getattr(E, k)) for k in "cbn"}
if a.nwin:   # a window read past the end stays inside
    code["n"] = np.concatenate([code["n"], np.zeros((1, a.nwin), np.uint32)], 1)
cbytes, bbytes, nbytes = (code[k].shape[1] * 4 for k in "cbn")
tbytes = 4 * (L if IMG is not None else pad16(tabarr).shape[1])   # the table width, gateway words included
nfull = nbytes
if a.nwin:
    nbytes = min(nbytes, a.nwin * 4)   # cb_ncode holds the window; BRISC loads the first one

CB_V, CB_CC, CB_BC_, CB_NC_, CB_TAB, CB_HALO, CB_SYNC, CB_STAGE = range(8)
CB_TOK, CB_CB, CB_BC, CB_CN, CB_NC, CB_BN, CB_NB, CB_DONE, CB_HDR = range(8, 17)
CB_IN, CB_OUT, CB_RS, CB_BS, CB_GC, CB_XR, CB_SH, CB_FH, CB_TS, CB_H, CB_OH = range(17, 28)
RING = a.ring
XB = 16 if m.red2 else 12 * PP   # all-gather block per chip (vm_brisc.cpp): its chip sum, or P partials per sum (up to 3)
KW = -(-K * P // 4) * 4   # partials stride, 16-B aligned (vm_brisc.cpp KW)
BW = 4 * (-(-max(K, P) // 4) * 4) + 16 if m.red2 else 8 * KW + 16   # broadcast region; room for a pair without red2
GB = 1024 if 1024 + 32 * P <= 4000 else 4096   # vm_brisc.cpp gather slots: before the fixed words at 4000, or after
RB = 4096 if GB == 1024 else GB + 32 * P       # reduction broadcast
E0 = RB + 2 * BW    # mesh words to E0 + 1616
if a.bpatch is None or a.bpatch < 0:
    lm = E.tab_stream.get("loc_mean", 0.0) if isinstance(getattr(E, "tab_stream", None), dict) else 0.0   # (images: pass --bpatch)
    a.bpatch = 4 if lm >= a.bpatch_min else 0
    print("--bpatch auto: %.1f local residue entries per patched map tile (min %.0f) -> %d" % (lm, a.bpatch_min, a.bpatch), flush=True)
assert 0 <= a.bpatch < 8 and not (a.bpatch and not a.tabstream), "--bpatch K: 1..7 with --tabstream"
assert 0 <= a.bgts < 8 and not (a.bgts and not a.tabstream), "--bgts K: 1..7 with --tabstream"
assert not a.tsort or a.tabstream, "--tsort 1 needs --tabstream"
assert 0 <= a.tgts < 8 and not (a.tgts and not a.tabstream) and a.tgts < (a.bgts or 8), "--tgts J: 1..7 with --tabstream, below --bgts"
# + the B-patch mailbox (vm_brisc.cpp bsvc) and the T-gather one before it (vm_compute.cpp tsvc)
SYNCB = -(-(E0 + 1616 + (256 if a.bpatch or a.bgts or a.tgts else 0) + (256 if a.tgts else 0)) // 4096) * 4096


def cbd(i, pages, page, fmt=ttnn.uint32):
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt,
                                                                         page_size=page)])


def out_cb(ring):
    """CB_OUT, plus bf16 views of the same memory under CB_H / CB_OH for --narrow: C_LDRH and C_STWH aim their
    pointers at CB_IN / CB_OUT pages themselves, so the views need a format and a page size but no L1 of their own."""
    fmts = [ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.float32, page_size=4096)]
    fmts += [ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048) for i in (CB_H, CB_OH)[:a.narrow]]
    return ttnn.CBDescriptor(total_size=ring * 4096, core_ranges=grid, format_descriptors=fmts)


TOKD = 64
cbs = [cbd(CB_V, nv, 4096, ttnn.float32), cbd(CB_CC, a.page, 1024) if a.page else cbd(CB_CC, 1, cbytes), cbd(CB_BC_, 1, bbytes), cbd(CB_NC_, 1, nbytes),
       cbd(CB_TAB, 1, tbytes), cbd(CB_HALO, 1, -(-m.hwords * 4 // 64) * 64), cbd(CB_SYNC, 1, SYNCB),
       cbd(CB_STAGE, 1, 16384)] + [cbd(i, TOKD, 32) for i in range(CB_TOK, CB_DONE)] + \
      [cbd(CB_DONE, 1, 32), cbd(CB_HDR, 1, 4096)]
if m.ndram:
    cbs += [cbd(CB_IN, RING, 4096 + DA, ttnn.float32), out_cb(RING), cbd(CB_RS, 1, 4096 + 16 * DA),
            cbd(CB_BS, 1, 16 * DA)] + ([cbd(CB_TS, 1, 2 * (4096 + 64 * 32))] if a.tabstream else [])
else:
    cbs += [cbd(CB_IN, 1, 64, ttnn.float32), cbd(CB_OUT, 1, 64, ttnn.float32), cbd(CB_RS, 1, 64), cbd(CB_BS, 1, 64)]
cbs += [cbd(CB_XR, 1, 512 + 2 * K * XB), cbd(CB_SH, 1, NSH * SHS * 4), cbd(CB_FH, 1, 512)]
if a.narrow and not m.ndram:
    cbs += [cbd(i, 1, 2048, ttnn.bfloat16) for i in (CB_H, CB_OH)[:a.narrow]]
if a.gcache < 0:   # at T 39 a 6-tile cache re-read each U-solve source ~30 times (k2 Small: 8x the k4 sweep time)
    a.gcache = max(0, min(m.T + 1, (L1_CB - sum(c.total_size for c in cbs)) // 4096))
    print("gcache %d slots (T %d)" % (a.gcache, m.T), flush=True)
cbs += [cbd(CB_GC, max(a.gcache, 1), 4096 if a.gcache else 64)]
l1 = sum(c.total_size for c in cbs)
print("L1 bytes per core %d of %d (sync %d, shadow %d)" % (l1, L1_CB, SYNCB, NSH * SHS * 4), flush=True)
assert l1 <= L1_CB, "CBs %d B per core, %d B over the %d B L1 CB region: shrink --gcache / --nwin / --ring" % (l1, l1 - L1_CB, L1_CB)
if a.l1_check:   # no upload: one-row DRAM tensors (an interleaved accessor's compile-time args carry no shape), so
    # the program built from them is the run's own, kernels, defines and runtime args alike; limit 0 ends every kernel at entry
    t_arena, t_c, t_b, t_n, t_tab, t_const, t_hist, t_out, t_home, t_blob = (to_dev(np.zeros((K, 1024), np.uint32)) for _ in range(10))
    TBP, a.limit = 1, 0
elif IMG is not None:   # chip by chip: chip k's arrays are built from its image files only while it is uploaded
    TBP = IMG.s["TBP"]
    STATE = None
    if a.state_in:
        with np.load(a.state_in) as z:     # read once: a key of an npz is re-read from the file on every access
            STATE = {k: z[k] for k in z.files if k != "hist"}
        print("state from %s" % a.state_in, flush=True)
        stamp("state read")
    up = lambda name: image_dev.upload(md, K, DEVBLK, lambda k: image_dev.chip_arrays(IMG, k, P, STATE, name)[name], to_dev)
    t_arena, t_const, t_home, t_blob = up("arena"), up("consts"), up("home"), up("blob")
    t_tab = image_dev.upload(md, K, DEVBLK, chip_tab, to_dev)
    stamp("images uploaded (arena, consts, home, blob, tab)")
    t_c, t_b, t_n = (to_dev(np.tile(code[k], (K, 1))) for k in "cbn")
    t_hist = to_dev(np.zeros((K * H, 16), np.uint32))
    t_out = to_dev(np.zeros((K * P, 1024 + nv * 1024), np.uint32))
else:
    arena = pad16(E.arena.reshape(K * P, -1))
    tabp = pad16(tabarr)
    HPG = max(m.hpages, 1)
    consts = per_chip(lambda k: E.consts[k * P:(k + 1) * P].reshape(-1, 1024))    # page core * cpages + offset + t
    home = per_chip(lambda k: E.home[k * P:(k + 1) * P].reshape(-1, 1024)).view(np.uint32)
    hmax = max(mp.hmax, 1)
    H = a.hist
    t_arena, t_tab, t_const = to_dev(arena), to_dev(tabp), to_dev(pad16(consts))
    t_c, t_b, t_n = (to_dev(np.tile(code[k], (K, 1))) for k in "cbn")
    t_hist = to_dev(np.zeros((K * H, 16), np.uint32))
    t_home = to_dev(home)
    t_out = to_dev(np.zeros((K * P, 1024 + nv * 1024), np.uint32))
    # streamed residue tables: each core's blob padded to whole 4 KB pages, page (local core) * TBP + k on its chip
    TBP = -(-E.tab_blob.shape[1] // 1024) if E.tab_stream else 1
    blob = np.zeros((K * P * TBP, 1024), np.uint32)
    if E.tab_stream:
        blob.reshape(K * P, TBP * 1024)[:, :E.tab_blob.shape[1]] = E.tab_blob
    t_blob = to_dev(blob)
defs = [("CFD_PROF", "1")] * a.prof + [("CFD_STOP", str(a.stop))] * bool(a.stop) + [("CFD_CP", str(m.cpages)), ("CFD_MESH", str(K)), ("CFD_NSH", str(NSH)), ("CFD_SYNCB", str(SYNCB)), ("CFD_GB", str(GB)), ("CFD_RB", str(RB)), ("CFD_CONS", str(E0 + 512))] + ([("CFD_RED2", "1")] if m.red2 else []) + \
       ([("CFD_TABSTREAM", str(CB_TS))] if a.tabstream else []) + ([("CFD_WIDE", "1")] if ttfluid.resident.encode.WIDE else []) + ([("CFD_BSCAL", "1")] if a.bscal else []) + ([("CFD_OS", str(a.os))] if a.os else [])


def kd(src, ct, cfg, rt=None, crt=None):
    k = ttnn.KernelDescriptor(kernel_source=str(DEV / src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=grid, compile_time_args=list(ct), config=cfg)
    if rt is not None:
        k.runtime_args = rt
    if crt is not None:
        k.common_runtime_args = list(crt)
    k.defines = defs + ([("CFD_ICACHE", "1")] if a.icache and src == "vm_compute.cpp" else []) + \
        ([("CFD_PHASE", "1")] if a.phase and src in ("vm_compute.cpp", "vm_brisc.cpp") else []) + \
        ([("CFD_TSORT", "1")] if a.tsort and src == "vm_ncrisc.cpp" else []) + \
        ([("CFD_BPATCH", str(a.bpatch))] if a.bpatch and src in ("vm_ncrisc.cpp", "vm_brisc.cpp") else []) + \
        ([("CFD_BGTS", str(a.bgts))] if a.bgts and src in ("vm_ncrisc.cpp", "vm_brisc.cpp") else []) + \
        ([("CFD_TGTS", str(a.tgts))] if a.tgts and src in ("vm_ncrisc.cpp", "vm_compute.cpp") else []) + \
        ([("CFD_GTILE", "1")] if a.gtile and src == "vm_ncrisc.cpp" else []) + \
        ([("CFD_NARROW", str(CB_H))] if a.narrow and src == "vm_compute.cpp" else []) + \
        ([("CFD_NARROW_OUT", str(CB_OH))] if a.narrow >= 2 and src == "vm_compute.cpp" else [])
    return k


acc = []
for t in (t_arena, t_c, t_b, t_n, t_tab, t_const, t_hist, t_out, t_home):
    acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
acc += [CB_OUT, CB_BS, m.T, m.hpages, RING, a.page, CB_XR, CB_SH, SHS, CB_FH, 0, 0, bbytes]   # BWIN 0: the whole b stream
nacc = [a.page, CB_CC]
for t in (t_const, t_home, t_c):
    nacc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
nacc += [cbytes, a.nwin, NMAX] + ttnn.TensorAccessorArgs(t_n).get_compile_time_args() + [nfull, CB_GC, a.gcache]
nacc += ttnn.TensorAccessorArgs(t_blob).get_compile_time_args()   # CFD_TABSTREAM: after the gather cache args
common = [t.buffer_address() for t in (t_arena, t_c, t_b, t_n, t_tab, t_const, t_hist, t_out)] + \
         [int(p0.x), int(p0.y), int(p1.x), int(p1.y), H, a.limit] + xyw + [t_home.buffer_address(), t_blob.buffer_address()]
# The B agent runs on BRISC with NoC 1 (multicast acks are reliable there, see run_vm.py --swap): on WH NCRISC
# executes from a 16 KB IRAM, which the fabric sender code overflows. The N agent runs on NCRISC with NoC 0.
_bn, _nn = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if a.bnoc == 1 else (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
BCFG = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=_bn)
NCFG = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=_nn)
DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}   # W E S N in mesh (row, col) steps
run_tag = [0]


def chip_program(rc):
    r, c = rc
    k = pid_at[rc]
    prog = ttnn.ProgramDescriptor()
    # semaphores and CBs first: setup_fabric_connection appends the connection's own semaphores to the program
    prog.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in (0, 1)]
    prog.cbs = cbs
    rt, nrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tail = [run_tag[0], k, c, C, r, R] + [pid_at[cells[q]] for q in range(K)]
    rows = {}
    for i, (x, y) in enumerate(cores):
        g = k * P + i
        args = [i, int(E.tab_halo[g]) | (1 << 31 if NPL > 1 else 0), tab_gw[g]] + tail
        if i == 0:
            for d in range(4):
                nb = (r + DIRS[d][0], c + DIRS[d][1])
                if nb in node:
                    args += list(ttnn.setup_fabric_connection(node[rc], node[nb], a.rlink, prog, ttnn.CoreCoord(x, y)))
        if g in gtab:
            peer_chip = chip(gtab[g]["peer"])
            nb = cells[place[peer_chip]]
            args += list(ttnn.setup_fabric_connection(node[rc], node[nb], LINKS[gtab[g]["lane"]], prog, ttnn.CoreCoord(x, y)))
        rows[(x, y)] = args
        nrt[x][y] = [i, int(E.tab_halo[g]) | (1 << 31 if NPL > 1 else 0), TBP]
    width = max(len(v) for v in rows.values())
    for (x, y), v in rows.items():
        rt[x][y] = v + [0] * (width - len(v))
    brisc = kd("vm_brisc.cpp", [CB_V, CB_CC, CB_BC_, CB_NC_, CB_TAB, CB_HALO, CB_SYNC, CB_STAGE, CB_CB, CB_BC, CB_BN,
                                CB_NB, CB_DONE, CB_HDR, nv, P, cbytes, bbytes, nbytes, tbytes, hmax] + acc,
               BCFG, rt=rt, crt=common)
    ncrisc = kd("vm_ncrisc.cpp", [CB_V, CB_NC_, CB_TAB, CB_HALO, CB_SYNC, CB_NC, CB_CN, CB_NB, CB_BN, CB_HDR, hmax, P,
                                 CB_IN, CB_RS, m.T, m.hpages, RING] + nacc,
                NCFG, rt=nrt, crt=common)
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
    um = [ttnn.UnpackToDestMode.Default] * NCB
    um[CB_V] = ttnn.UnpackToDestMode.UnpackToDestFp32
    um[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = um
    comp = kd("vm_compute.cpp", [CB_V, CB_CC, CB_TOK, CB_CB, CB_BC, CB_CN, CB_NC, CB_DONE, CB_HDR, nv, CB_SYNC, CB_IN,
                                 CB_OUT, a.page], cc, crt=[a.limit])
    prog.kernels = [brisc, ncrisc, comp]
    return prog


class Aiclk:
    """tt_aiclk of every chip of the mesh every 20 ms (.110: UMD 0-15 are nodes 16-31, 16-23 8-15, 24-31 0-7; another
    host names the sysfs node of each visible chip in CFD_TT_NODES, e.g. qb1 logical 0,1 are nodes 1,2)."""
    NODE = {u: (u + 16 if u < 16 else (u - 8 if u < 24 else u - 24)) for u in range(32)}

    def __enter__(self):
        nodes = os.environ.get("CFD_TT_NODES")
        nodes = [int(x) for x in nodes.split(",")] if nodes else [self.NODE[u] for u in umd]
        self.paths = [f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk" for n in nodes]
        self.vals, self.stop = [], threading.Event()

        def loop():
            while not self.stop.is_set():
                for p in self.paths:
                    try:
                        v = int(open(p).read().split()[0])
                        if 0 < v < 5000:
                            self.vals.append(v)
                    except Exception:
                        pass
                time.sleep(0.02)
        self.th = threading.Thread(target=loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *x):
        self.stop.set()
        self.th.join()

    def summary(self):
        return dict(min=min(self.vals), max=max(self.vals), n=len(self.vals)) if self.vals else dict(min=None, max=None, n=0)


def main():
    global t_home, t_arena
    io = [t_arena, t_c, t_b, t_n, t_tab, t_const, t_hist, t_out, t_home, t_blob]
    walls, ticks, clks = [], [], []
    o = None
    assert not a.legs or IMG is not None, "--legs: with --from-images"
    assert not a.cases or (IMG is not None and not a.legs), "--cases: with --from-images, without --legs"
    if a.l1_check:
        return l1_check(io)
    if OVL:
        return main_overlap(io)
    rows_all = []
    nxt = None   # --cases: the next case's homes, built on the host during this case's run
    for rr in range(a.cases or a.legs or a.runs):
        if nxt is not None:   # written into the same buffer: the program's addresses stay valid
            ttnn.copy_host_to_device_tensor(nxt, t_home)
            nxt = None
            stamp("case %d: homes written" % rr)
        elif rr and a.legs:   # the next leg continues: homes stay on the device, the arena's state registers from the dump
            ttnn.deallocate(t_arena)
            t_arena = image_dev.upload(md, K, DEVBLK, lambda k: image_dev.carry_arena(
                IMG, k, P, o[k * P:(k + 1) * P, 1024:].reshape(P, nv, 1024)), to_dev)
            io[0] = t_arena
            common[0] = t_arena.buffer_address()
            stamp("leg %d: arena carried" % rr)
        elif rr:   # every run starts from the uploaded state: the DRAM homes were written by the last one
            ttnn.deallocate(t_home)
            t_home = up("home") if IMG is not None else to_dev(home)
            io[-1] = t_home
            common[-1] = t_home.buffer_address()
        launch(rr, io)
        with Aiclk() as clk:
            w0 = time.time()
            ttnn.generic_op(io, mesh_program())
            print("run %d: enqueued after %.1f s (compile included), waiting for the device" % (rr, time.time() - w0), flush=True)
            if rr + 1 < a.cases:
                nxt = next_homes()
                stamp("case %d: homes built on the host" % (rr + 1))
            ttnn.synchronize_device(md)
            walls.append(time.time() - w0)
        clks.append(clk.summary())
        o = from_dev(t_out, np.uint32)
        o = o.reshape(K * P, -1)
        ticks.append(core0_ticks(o))
        print("run %d wall %.3f s core0 ticks %s aiclk %s" % (rr, walls[-1], ticks[-1], clks[-1]), flush=True)
        stamp("run %d done, dump read" % rr)
        if a.legs:   # every leg's history rows, in order
            hl = from_dev(t_hist, np.uint32).reshape(K, H, 16)
            rows_all += [{nm: float(hl[0, q, i:i + 1].view(np.float32)[0]) for i, nm in enumerate(E.out_names)}
                         for q in range(min(int(o.reshape(K * P, -1)[0, 14]), H))]
        if a.cases:   # every case its own answer, read before the next case's homes overwrite this one's
            report(o, walls[-1:], ticks[-1:], clks[-1:], None, ".c%d" % rr)
    if not a.cases:
        report(o, walls, ticks, clks, rows_all if a.legs else None, "")


def l1_check(io):
    """The kernel config buffer (binaries + runtime args + CB and semaphore config per core) is sized by metal when the
    program is enqueued, so the check enqueues it: limit 0 ends every kernel at entry, nothing runs."""
    try:
        ttnn.generic_op(io, mesh_program())
        ttnn.synchronize_device(md)
    except RuntimeError as e:
        over = re.search(r"Program size \((\d+)\) too large for kernel config buffer \((\d+)\)", str(e))
        if not over:
            raise
        size, cap = map(int, over.groups())
        print("--l1-check: config buffer: over by %d B (program %d, buffer %d)%s" % (
            size - cap, size, cap, ", use --os %d" % (a.os + 1) if a.os < 2 else ""), flush=True)
        raise SystemExit(4)
    print("--l1-check: fits, %d B L1 spare, config buffer OK at --os %d" % (L1_CB - l1, a.os), flush=True)


def launch(rr, io):
    run_tag[0] += 1
    ttnn.synchronize_device(md)
    print("run %d: program built, launching" % rr, flush=True)


def mesh_program():
    mpd = ttnn.MeshProgramDescriptor()
    for rc in cells:
        mc = ttnn.MeshCoordinate(*rc)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = chip_program(rc)
    return mpd


def next_homes():   # the next case's homes as one multi-device host tensor (the image's start or --state-in)
    return image_dev.host_shards(md, K, DEVBLK, lambda k: image_dev.chip_arrays(IMG, k, P, STATE, "home")["home"])[0]


def core0_ticks(o):
    return [int(o[k * P, 10]) + (int(o[k * P, 58]) << 32) for k in range(K)]


def main_overlap(io):
    """--cases with --overlap 1. Every buffer the program addresses stays where it is; between two cases, on CQ 0 and
    only while no program runs, the case's dump, history and state home pages are copied into fresh device tensors and
    the next case's homes are copied in from a staging tensor, then the next case launches. CQ 1 only moves data
    between the host and DRAM tensors no program uses (never a device op: the program holds every worker core): it
    writes the staging homes during a case and reads the previous case's copies, so the host reads and saves case k
    while case k+1 runs. Per-case states equal the serial path's: the copies are exact, the program is the same."""
    Q0, Q1 = ttnn.QueueId(0), ttnn.QueueId(1)
    pr = image_dev.state_pages(IMG)
    stage = ttnn.allocate_tensor_on_device(t_home.shape, t_home.dtype, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    staged = nxt = None   # CQ 1's event after the staging write of the next case's homes, and those homes on the host
    prev = None     # the previous case's device copies + its wall/clock, read on CQ 1 while this case runs
    for rr in range(a.cases):
        if rr:
            ttnn.wait_for_event(Q0, staged)
            try:
                ttnn.copy(stage, t_home)
                stamp("case %d: homes copied from the stage" % rr)
            except Exception as e:   # a ttnn whose copy op does not take this tensor: the serial path's host write
                print("stage copy failed (%s), writing the homes from the host" % str(e).splitlines()[0][:120], flush=True)
                ttnn.copy_host_to_device_tensor(nxt, t_home)
                stamp("case %d: homes written" % rr)
            nxt = None
        copied = ttnn.record_event(md, cq_id=Q0)   # the stage is free for CQ 1 once this point of CQ 0 has run
        launch(rr, io)
        with Aiclk() as clk:
            w0 = time.time()
            ttnn.generic_op(io, mesh_program())
            print("run %d: enqueued after %.1f s (compile included), waiting for the device" % (rr, time.time() - w0), flush=True)
            if prev is not None:   # case rr-1's answer, read over CQ 1 while case rr runs
                report_copies(*prev)
                prev = None
            if rr + 1 < a.cases:
                nxt = next_homes()
                stamp("case %d: homes built on the host" % (rr + 1))
                ttnn.wait_for_event(Q1, copied)   # the stage is free once the copy out of it ran
                ttnn.copy_host_to_device_tensor(nxt, stage, cq_id=Q1)
                staged = ttnn.record_event(md, cq_id=Q1)
                stamp("case %d: homes staged" % (rr + 1))
            ttnn.synchronize_device(md, Q0)
            wall = time.time() - w0
        # case rr's outputs out of the buffers the next case overwrites, on the device (the CQ 0 copies run before the next
        # launch; CQ 1 reads them after this event)
        home3 = ttnn.reshape(t_home, ttnn.Shape([P, HPG, image_dev.LANES]))   # each device's homes as (core, page, lane)
        # the dump is read here on CQ 0 (~1.3 s): a row-major clone stages whole rows in L1 and t_out's (1 + nv) * 4 KB
        # rows do not fit (1.58 MB on Small k32); the history and the state pages have narrow rows and are copied
        snap = (ttnn.from_device(t_out), ttnn.clone(t_hist), [ttnn.slice(home3, [0, lo, 0], [P, hi, image_dev.LANES]) for lo, hi in pr])
        done = ttnn.record_event(md, cq_id=Q0)
        stamp("run %d done (wall %.3f s), outputs copied on the device" % (rr, wall))
        prev = (rr, snap, done, pr, wall, clk.summary())
    report_copies(*prev)
    ttnn.deallocate(stage)


def report_copies(rr, snap, done, pr, wall, clk):
    Q1 = ttnn.QueueId(1)
    ttnn.wait_for_event(Q1, done)
    t_o, t_h, pages = snap
    rd = lambda t: ttnn.from_device(t, queue_id=Q1)
    o = mesh_np(t_o, np.uint32).reshape(K * P, -1)   # already on the host
    hist = mesh_np(rd(t_h), np.uint32)
    hp = [ttnn.to_torch(rd(t), mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(np.uint32) for t in pages]
    for t in (t_h, *pages):
        ttnn.deallocate(t)
    stamp("case %d: outputs read on CQ 1" % rr)

    def home_pages(d):   # device d's homes with only the state page ranges filled, as download_home_pages
        out = np.zeros((P, HPG, image_dev.LANES), np.uint32)
        for (lo, hi), h in zip(pr, hp):
            out[:, lo:hi] = h[d * P:(d + 1) * P]
        return out
    tk = core0_ticks(o)
    print("run %d wall %.3f s core0 ticks %s aiclk %s" % (rr, wall, tk, clk), flush=True)
    report(o, [wall], [tk], [clk], None, ".c%d" % rr, hist=hist, home_pages=home_pages)


def mesh_np(h, dt):   # a multi-device HOST tensor back in block order (from_dev without the device read)
    x = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(dt)
    return x.reshape(K, -1, *x.shape[1:])[list(place)].reshape(x.shape)


def report(o, walls, ticks, clks, rows_legs, tag, hist=None, home_pages=None):
    hdr = o[:, :1024]
    hist_all = from_dev(t_hist, np.uint32) if hist is None else hist
    hist_all = hist_all.reshape(K, H, 16)
    nrow = int(hdr[0, 14])
    rows = [{nm: float(hist_all[0, q, i:i + 1].view(np.float32)[0]) for i, nm in enumerate(E.out_names)}
            for q in range(min(nrow, H))]
    if rows_legs is not None:
        rows, nrow = rows_legs, len(rows_legs)
    if IMG is not None:   # chip by chip: chip k's DRAM homes come back from the device that ran block k
        dev_of = {DEVBLK[d]: d for d in range(K)}
        pr = image_dev.state_pages(IMG)
        dev_state = IMG.state(lambda k: o[k * P:(k + 1) * P, 1024:].reshape(P, nv, 1024).view(np.float32),
                              lambda k: (home_pages or (lambda d: image_dev.download_home_pages(t_home, d, P, HPG, pr)))(dev_of[k]).view(np.float32))
        print("state: %d of %d home pages read, ranges %s" % (sum(h - l for l, h in pr), HPG, pr), flush=True)
        stamp("state downloaded")
    else:
        home_out = from_dev(t_home, np.float32)
        home_out = home_out.reshape(K * P, HPG, 1024)
        dev_state = E.state_from_arena(o[:, 1024:].reshape(K * P, nv, 1024).view(np.float32), home_out)
    amax = max(c["max"] or 0 for c in clks) or 1000
    h = hdr.astype(np.int64)
    for w in (48, 50, 51, 52, 53, 54, 56):   # MESH_ACC's carry word is w + 400
        h[:, w] += h[:, w + 400] << 32
    root = [k * P for k in range(K)]
    gws = sorted(gtab)
    res = dict(jointred=a.jointred, cg1=a.cg1, fold1=a.fold1, fusepart=a.fusepart, gsplit=a.gsplit, wplace=a.wplace, bcast=a.bcast, ring=a.ring, page=a.page, nwin=a.nwin, sched=a.sched, dedup=a.dedup, tsort=a.tsort, bpatch=a.bpatch, bgts=a.bgts, tgts=a.tgts, gtile=a.gtile, amg=a.amg, amg_depth=a.amg_depth, amg_min_core=a.amg_min_core, hoist=a.hoist, tabstream=a.tabstream, state_in=a.state_in, gcache=a.gcache, icache=a.icache, red2=m.red2, redm=m.redm, fuse=a.fuse, what=a.what, n=a.n, chips=umd, mesh=[R, C], K=K, layout=a.layout, T=m.T, l1_vec=a.l1, ndram=m.ndram, narrow=a.narrow, case=a.case,
               plan=info, sizes=E.sizes(), l1=l1, rows=nrow, walls=walls, ticks=ticks,
               ms_device=[max(t) / amax / 1e3 for t in ticks], aiclk=clks,
               same_history_all_chips=bool(all(np.array_equal(hist_all[0, :nrow], hist_all[k, :nrow]) for k in range(K))),
               all_cores_done=bool(np.all(hdr[:, 18] == 0xB0DE0000)),
               abort=dict(cores=int(np.sum(hdr[:, 34] != 0)),
                          brisc=sorted({(int(x), int(q)) for x, q in hdr[:, 30:32] if x}),
                          ncrisc=sorted({(int(x), int(q)) for x, q in hdr[:, 32:34] if x}),
                          # per chip: {reason: cores} for BRISC / NCRISC, and each gateway's own reason
                          chips={k: [dict(sorted(collections.Counter(int(x) for x in hdr[k * P:(k + 1) * P, c] if x).items()))
                                     for c in (30, 32)] for k in range(K)},
                          gw={int(g): int(hdr[g, 30]) for g in gws}),
               counts=dict(red=int(hdr[0, 13]), halo=int(hdr[0, 17]), rows=nrow),
               stage=dict(root=[int(h[q, 57]) for q in root], gw={int(g): int(h[g, 57]) for g in gws}),
               mesh_cyc=dict(barrier=[int(h[q, 56]) for q in root], xred=[int(h[q, 48]) for q in root],
                             xred_n=[int(h[q, 49]) for q in root],
                             halo_all_max=int(h[:, 54].max()), halo_all_med=float(np.median(h[:, 54])),
                             gw={int(g): dict(relay=int(h[g, 50]), members=int(h[g, 51]), rcred=int(h[g, 52]),
                                              send=int(h[g, 53])) for g in gws}),
               last=rows[-1] if rows else None)
    if a.phase:   # hdr 512 + 2 id MATH, 544 BRISC, 576 B_RED, 608 B_HALO, 640 B_WAITC (vm_compute / vm_brisc CFD_PHASE)
        from ttfluid.resident.program import PHASES
        hz = amax * 1e6
        w64 = lambda base, i: o[:, base + 2 * i].astype(np.float64) + o[:, base + 2 * i + 1].astype(np.float64) * 2.0 ** 32
        tot = sum(w64(512, i) for i in range(16))
        ph = {}
        for i, nm in list(enumerate(PHASES)) + [(15, "init")]:
            x = w64(512, i) / hz
            if not x.any():
                continue
            st = lambda v: dict(min=float(v.min()), med=float(np.median(v)), max=float(v.max()))
            ph[nm] = dict(**st(x), share=float(np.median(w64(512, i) / np.maximum(tot, 1))),
                          brisc=st(w64(544, i) / hz), red=st(w64(576, i) / hz), halo=st(w64(608, i) / hz), waitc=st(w64(640, i) / hz))
            print("phase %-12s math s min %.4f med %.4f max %.4f share %5.1f%% | brisc red med %.4f halo med %.4f waitc med %.4f"
                  % (nm, x.min(), np.median(x), x.max(), 100 * ph[nm]["share"], ph[nm]["red"]["med"], ph[nm]["halo"]["med"],
                     ph[nm]["waitc"]["med"]), flush=True)
        res["phase"] = ph
    if a.prof:
        from ttfluid.resident.census import prof
        res["census"] = prof(hdr)
    if ref is not None:
        eh, es = ref
        worst = 0.0
        for x, y in zip(eh, rows):
            for kk in x:
                if not kk.startswith("its"):
                    worst = max(worst, abs(x[kk] - y[kk]) / max(abs(x[kk]), 1e-30))
        res["vs_emu"] = dict(rows=(len(eh), len(rows)), worst_rel=worst,
                             its_equal=all(x.get(kk) == y.get(kk) for x, y in zip(eh, rows) for kk in x if kk.startswith("its")),
                             state={kk: float(np.max(np.abs(np.asarray(es[kk], np.float64) - dev_state[kk])) /
                                              max(float(np.max(np.abs(es[kk]))), 1e-30)) for kk in es if np.size(es[kk])})
    if tag:
        res["case_index"] = int(tag[2:])
    np.savez(str(Path(a.out).with_suffix("")) + f"{tag}_{a.what}_{R}x{C}_state.npz",
             **{kk: np.asarray(v) for kk, v in dev_state.items()}, hist=json.dumps(rows))
    stamp("state saved")
    if a.write and a.what == "case" and not res["abort"]["cores"]:
        import shutil
        from ttfluid.resident.program import apply_state
        out = Path(a.write)
        if not out.exists():
            out.mkdir(parents=True)
            for d in ("0", "constant", "system"):
                shutil.copytree(Path(a.case) / d, out / d)
        s2 = SimpleFoam(a.case, backend="ell", dtype=np.float64, log=lambda *x: None)
        apply_state(s2, {kk: np.asarray(v, np.float64) for kk, v in dev_state.items()}, nrow)
        s2.write(out)
        res["written"] = f"{out}/{nrow}"
    print(json.dumps(res), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    if res["abort"]["cores"] or not nrow:   # a run that aborted on its bounded waits is a failure, not a row
        raise SystemExit(3)


try:
    main()
finally:   # a failed build or run must still release the mesh, or the interpreter hangs at exit
    ttnn.close_mesh_device(md)
    if FAB:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
