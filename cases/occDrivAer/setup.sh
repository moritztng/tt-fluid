#!/bin/bash
# Build the 1,606,238-cell occDrivAer case from the OpenFOAM HPC committee repository, ready to lower with tt-fluid.
# Needs OpenFOAM v2412 in the environment, git, about 40 GB of free RAM for snappyHexMesh, and NP cores.
#   cases/occDrivAer/setup.sh DIR [NP] [SHIFT]  ->  DIR/occDrivAer (constant/polyMesh, 0/ from potentialFoam, system/)
# SHIFT (default 4) is how many refinement levels below the committee's dicts to mesh; 3 gives a finer mesh.
set -euo pipefail
DIR=${1:?usage: setup.sh DIR [NP] [SHIFT]}; NP=${2:-8}; SHIFT=${3:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=https://develop.openfoam.com/committees/hpc.git
SHA=84c262431117f5c921db8335e368a12d0e9fa3f0
STATIC=incompressible/simpleFoam/occDrivAerStaticMesh
ROT=incompressible/pimpleFoam/LES/occDrivAerRotMesh
mkdir -p "$DIR"; DIR=$(cd "$DIR" && pwd)
log() { echo "$(date -u +%FT%TZ) $*"; }
run() { local n=$1; shift; log "$n"; "$@" > "logs/$n.log" 2>&1 || { log "FAIL $n, see $PWD/logs/$n.log"; exit 1; }; }

# 1. Upstream dicts and geometry, pinned
if [ ! -d "$DIR/hpc/.git" ]; then
  git clone -q --filter=blob:none --no-checkout "$REPO" "$DIR/hpc"
  git -C "$DIR/hpc" sparse-checkout set --no-cone "$STATIC" "$ROT"
fi
git -C "$DIR/hpc" checkout -q "$SHA"

# 2. Coarse mesh: the RotMesh snappy dicts, four refinement levels down, one static region
rm -rf "$DIR/mesh"
python3 "$HERE/coarsen.py" "$DIR/hpc/$ROT/occDrivAerRotMesh.orig" "$DIR/mesh" "$SHIFT" "$NP"
cd "$DIR/mesh"; mkdir -p logs
cp system/fvSchemes.mesh system/fvSchemes
run blockMesh blockMesh
run surfaceFeatureExtract surfaceFeatureExtract
run decomposePar decomposePar -force
run snappyHexMesh mpirun -np "$NP" snappyHexMesh -overwrite -parallel
run reconstructParMesh reconstructParMesh -constant
run checkMesh checkMesh -constant
grep -E '^\s+cells:' logs/checkMesh.log

# 3. Run case: the static case's own dicts (fixed-tolerance solver controls) on the coarse mesh
C=$DIR/occDrivAer; rm -rf "$C"; mkdir -p "$C/logs"
cp -r "$DIR/hpc/$STATIC/system" "$DIR/hpc/$STATIC/0.orig" "$C/"
mkdir "$C/constant"; cp "$DIR/hpc/$STATIC"/constant/*Properties "$C/constant/"
cp -r "$DIR/mesh/constant/polyMesh" "$C/constant/"
cd "$C"
cp system/fvSolution.fixedTol system/fvSolution
sed -i -e "s/^nCores .*/nCores              $NP;/" -e "s/^decompositionMethod .*/decompositionMethod scotch;/" \
  system/include/caseDefinition

# 4. Initial state, as the case's Allrun: potentialFoam, then a boundary layer in U. The potential solve is converged
#    fully (the reference Cd history starts from this state), not to the case's relTol 0.01.
cp -r 0.orig 0
run decomposePar decomposePar -force
cp system/fvSolution system/fvSolution.run
for e in tolerance:1e-10 relTol:0 maxIter:5000; do foamDictionary -entry "solvers.Phi.${e%%:*}" -set "${e#*:}" system/fvSolution >/dev/null; done
run potentialFoam mpirun -np "$NP" potentialFoam -initialiseUBCs -parallel
mv system/fvSolution.run system/fvSolution
run applyBoundaryLayer mpirun -np "$NP" applyBoundaryLayer -ybl 0.0450244 -parallel
rm -rf 0; run reconstructPar reconstructPar -withZero -time 0
rm -rf processor*
log "done: $C"
