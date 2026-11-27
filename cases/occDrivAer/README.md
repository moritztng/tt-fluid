# occDrivAer, 1.6 M cells

The open-closed-cooling DrivAer car (Ford's notchback DrivAer with closed coolings and a detailed underbody) from the
OpenFOAM HPC Technical Committee's `incompressible/simpleFoam/occDrivAerStaticMesh` case: steady RANS, k-omega SST,
38.889 m/s, 47 car wall patches with wall functions, slip ground, Cd and Cl from the case's own `forceCoeffsAll`.

The committee's mesh has 65 M cells. `setup.sh` builds a 1,606,238-cell version of the same geometry instead, so the
reference can be computed with serial OpenFOAM on one workstation. Every boundary condition, scheme and solver setting is
the 65 M case's own (`fvSolution.fixedTol`); only the mesh is coarser.

## Build it

Needs OpenFOAM v2412 in the environment, git, and about 40 GB of free RAM for snappyHexMesh.

```bash
cases/occDrivAer/setup.sh work 8     # 8 MPI ranks; writes work/occDrivAer in about 4 minutes
```

The script clones the committee repository at a pinned commit, writes a snappyHexMesh case from the
`occDrivAerRotMesh` meshing dicts with every refinement level four steps coarser (`coarsen.py` lists each change),
meshes it, copies the static case's dicts onto that mesh, and initialises the fields as the case's `Allrun` does
(`potentialFoam -initialiseUBCs`, solved to 1e-10, then `applyBoundaryLayer`). With 8 ranks the result is bit-identical, mesh and
fields, to the starting state of the reference OpenFOAM run. Lower and run `work/occDrivAer` like the motorbike case in
the [tutorial](../../docs/tutorial.md); a single Blackhole chip holds it.

Cd from 3999 device iterations matches an 8-rank OpenFOAM run to 0.5 counts and a serial one to 15 counts, which is
how far the two OpenFOAM runs are apart ([accuracy](../../docs/accuracy.md#4-drag-and-lift-against-openfoam-occdrivaer-coarse-mesh)).

## Licence

The case dicts and geometry come from the HPC committee repository, which publishes them under CC BY-SA 4.0 (Upstream
CFD GmbH, exaFOAM project). `setup.sh` downloads them; nothing from that repository is stored here. `setup.sh` and
`coarsen.py` are part of tt-fluid (MIT).
