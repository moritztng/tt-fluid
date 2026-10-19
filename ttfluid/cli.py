"""ttfluid: run an OpenFOAM simpleFoam case on Tenstorrent chips, or on the CPU emulator of the same program.

    ttfluid run CASE [-n ITERS] [--arch bh|wh] [--chips K] [--write DIR] [-- RUNNER FLAGS]
    ttfluid check CASE [-n ITERS] [--arch bh|wh] [--chips K]

Both commands lower the case the same way and hand it to ttfluid.device.run_mesh (every runner flag passes through
after `--`). `run` executes on the device and, unless --no-check, first runs the CPU emulator, which is the exact
reference of the device program (close to the device, not bitwise: the chip approximates division and the elementary functions). `check` runs the emulator only and needs no card.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

from ttfluid import __version__

# Lowering flags for a case that fits in the chips' SRAM (the pitzDaily class). Jacobi-preconditioned PCG for p
# (--amg --amg-depth 0 builds no coarse level), fused, sliced operators, RCB partition.
SRAM = ["--layout", "rcb", "--amg", "--amg-depth", "0", "--amg-stack", "--amg-vec", "--amg-smoothed", "0", "--lsched",
        "--phase", "--sliced", "--fuse", "--cse", "1", "--hlate", "1"]
P300_SUBSYSTEMS = {"0x0044", "0x0045", "0x0046"}


def mesh_shape(chips):
    return {1: "1,1", 2: "1,2", 4: "2,2", 8: "2,4", 16: "4,4", 32: "4,8"}[chips]


def _p300_nodes():
    out = []
    for e in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            if (e / "device" / "subsystem_device").read_text().strip().lower() in P300_SUBSYSTEMS:
                out.append(int(e.name.rsplit("!", 1)[1]))
        except (OSError, ValueError):
            pass
    return out


def _mesh_descriptor(chips):
    """A p300 board is a custom topology for ttnn: one chip opens as a p150, both chips as the p300 mesh."""
    if os.environ.get("TT_MESH_GRAPH_DESC_PATH") or not _p300_nodes() or chips > 2:
        return
    import importlib.util
    spec = importlib.util.find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        return
    name = "p150_mesh_graph_descriptor.textproto" if chips == 1 else "p300_mesh_graph_descriptor.textproto"
    d = Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
    if d.is_file():
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(d)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="ttfluid", description=__doc__.split("\n\n")[0])
    ap.add_argument("-V", "--version", action="version", version=f"tt-fluid {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "run a case on Tenstorrent chips"), ("check", "run a case on the CPU emulator only")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("case", help="OpenFOAM case directory (constant/polyMesh, 0/, system/)")
        p.add_argument("-n", "--iterations", type=int, default=100, help="SIMPLE iterations, at most (residualControl "
                       "in fvSolution can stop earlier)")
        p.add_argument("--arch", default="bh", choices=["bh", "wh"], help="bh: Blackhole p150 / p300; wh: Wormhole")
        p.add_argument("--chips", type=int, default=1, choices=[1, 2, 4, 8, 16, 32])
        p.add_argument("--out", default="ttfluid_runs.jsonl", help="append one JSON result row per run here")
        if name == "run":
            p.add_argument("--write", default="", metavar="DIR", help="write the final U, p, k, omega, nut into a copy "
                           "of the case at DIR")
            p.add_argument("--no-check", action="store_true", help="skip the CPU emulator reference")
    a = ap.parse_args(argv)
    case = Path(a.case).resolve()
    if not (case / "constant" / "polyMesh").is_dir():
        ap.error(f"{case} has no constant/polyMesh")
    r = ["case", str(a.iterations), "--case", str(case), "--arch", a.arch, "--mesh", mesh_shape(a.chips),
         "--out", a.out, "--runs", "1", *SRAM]
    if a.chips == 2 and a.arch == "bh":
        # a p300 board links its two chips with Ethernet links 0 and 1: halos on 0, the reductions on 1
        r += ["--gw", "1", "--links", "0", "--rlink", "1"]
    if a.cmd == "check":
        r.append("--emuonly")
    else:
        if a.write:
            r += ["--write", str(Path(a.write).resolve())]
        if a.no_check:
            r.append("--noemu")
        os.environ.setdefault("TT_VISIBLE_DEVICES", ",".join(str(i) for i in range(a.chips)))
        if a.arch == "bh":
            _mesh_descriptor(a.chips)
    sys.argv = ["ttfluid.device.run_mesh", *r, *extra]
    runpy.run_module("ttfluid.device.run_mesh", run_name="__main__")


if __name__ == "__main__":
    main()
