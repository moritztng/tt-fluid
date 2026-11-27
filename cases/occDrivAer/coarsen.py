#!/usr/bin/env python3
"""Write a coarse, single-region occDrivAer snappyHexMesh case from the OpenFOAM HPC committee's occDrivAerRotMesh dicts.

The 65 M-cell occDrivAerStaticMesh case ships its mesh but not its meshing dicts. occDrivAerRotMesh meshes the same
geometry (OC_DrivAer_CC_NB_210512_A) with the same region names, so this script takes its snappyHexMeshDict.full and:
  - drops the AMI_* rotating-zone surfaces (the static case meshes the wheels in place),
  - renames the ground_layer plate to ground_noSlip_layer and adds an upstream ground_noSlip plate, so the patch set
    matches the static case's 55 patches,
  - lowers every castellated refinement level by SHIFT (setup.sh uses 4: 1,606,238 cells),
  - removes blockLevel (snappyHexMesh v2412 segfaults in removeGapCells with it on this geometry).

usage: coarsen.py ROTMESH_ORIG_DIR DST_CASE [SHIFT] [NPROC] [VSHIFT]
VSHIFT (default SHIFT) shifts the box refinement regions (levels ((1E15 n))) separately from the surfaces.
"""
import re
import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
shift = int(sys.argv[3]) if len(sys.argv) > 3 else 3
nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8

dst.mkdir(parents=True, exist_ok=False)
for d in ("system", "constant"):
    shutil.copytree(src / d, dst / d)
tri = dst / "constant" / "triSurface"
parts = sorted(tri.glob("OC_DrivAer_CC_NB_210512_A.obj.gz_a?"))
with open(tri / "OC_DrivAer_CC_NB_210512_A.obj.gz", "wb") as out:
    for p in parts:
        out.write(p.read_bytes())
for p in parts:
    p.unlink()
for p in tri.glob("AMI_*"):
    p.unlink()

sd = (dst / "system" / "snappyHexMeshDict.full").read_text()


def drop_block(text, name_re):
    # remove '<name> { ... }' blocks (one nesting level of braces inside allowed)
    pat = re.compile(r'\n\s*"?' + name_re + r'"?\s*\n\s*\{(?:[^{}]|\{[^{}]*\})*\}', re.S)
    return pat.sub("", text)


sd = drop_block(sd, r"AMI_[A-Z_]+\.obj")
sd = drop_block(sd, r"AMI_OUT\.\*")
sd = drop_block(sd, r"AMI_IN\.\*")
sd = re.sub(r'\n\s*\{\s*file\s+"[^"]*AMI[^"]*";[^{}]*\}', "", sd)
sfe = dst / "system" / "surfaceFeatureExtractDict"
sfe.write_text(drop_block(sfe.read_text(), r"AMI_[A-Z_]+\.obj"))
assert "AMI" not in sfe.read_text()
assert "AMI" not in sd, [l for l in sd.splitlines() if "AMI" in l]

sd = sd.replace("ground_layer", "ground_noSlip_layer")
sd = sd.replace('"(ground|ground_noSlip_layer)"', '"(ground|ground_noSlip|ground_noSlip_layer)"')
plate = """    ground_noSlip
    {
        type            searchablePlate;
        origin          (-40 -22 -0.3176);
        span            (30 44 0);
    }
    ground_noSlip_layer
"""
sd = sd.replace("    ground_noSlip_layer\n", plate, 1)

a = sd.index("castellatedMeshControls")
b = sd.index("snapControls")
cmc = sd[a:b]


def lv(n):
    return str(max(0, int(n) - shift))


cmc = re.sub(r"(\blevel\s+)(\d+)(\s*;)", lambda m: m.group(1) + lv(m.group(2)) + m.group(3), cmc)
cmc = re.sub(r"(\blevel\s*\(\s*)(\d+)(\s+)(\d+)(\s*\))",
             lambda m: m.group(1) + lv(m.group(2)) + m.group(3) + lv(m.group(4)) + m.group(5), cmc)
cmc = re.sub(r"(\bblockLevel\s+)(\d+)", lambda m: m.group(1) + lv(m.group(2)), cmc)
cmc = re.sub(r"(\blevels\s*)(\((?:\s*\([^()]*\))+\s*\))",
             lambda m: m.group(1) + re.sub(r"\(\s*([0-9.eE+-]+)\s+(\d+)\s*\)",
                                           lambda t: "(" + t.group(1) + " " + lv(t.group(2)) + ")",
                                           m.group(2)), cmc)
# v2412 snappyHexMesh segfaults in surfaceProximityBlock -> removeGapCells -> markFakeGapRefinement
# with blockLevel on this geometry (rc 139, 2026-10-10); gap blocking only removes cells, not patches.
cmc = re.sub(r"\n\s*blockLevel\s+\d+\s*;", "", cmc)
sd = sd[:a] + cmc + sd[b:]
(dst / "system" / "snappyHexMeshDict").write_text(sd)

# Below SHIFT 4 the upstream averageNeighbour geometry scheme divides by zero while snapping (FPE in rank 2, SHIFT 3,
# 2026-11-27); basic geometry only changes snappy's motion-quality metrics. SHIFT 4 keeps the upstream scheme so the
# 1,606,238-cell mesh stays reproducible.
if shift < 4:
    for f in ("fvSchemes", "fvSchemes.mesh"):
        q = dst / "system" / f
        q.write_text(re.sub(r"(\bgeometry\s*\{[^{}]*\btype\s+)averageNeighbour", r"\1basic", q.read_text()))
(dst / "system" / "decomposeParDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {nproc};
method scotch;
""")
(dst / "system" / "controlDict").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam; startFrom latestTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat binary; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
""")
print("wrote", dst, "shift", shift, "nproc", nproc)
