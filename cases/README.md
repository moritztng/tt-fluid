# Cases

`pitzDaily/` is OpenFOAM's `incompressible/simpleFoam/pitzDaily` tutorial from OpenFOAM v2412 (GPL-3.0-or-later,
copyright OpenCFD Ltd) with one change, `RASModel kOmegaSST` in `constant/turbulenceProperties` (the tutorial selects
kEpsilon; its `0/` already has the omega field SST needs), and with the mesh `blockMesh` writes already in `constant/polyMesh` (12,225 cells), so
it runs without an OpenFOAM installation. It is a 2D backward-facing step at Re ~ 25,000, solved with k-omega SST.

`occDrivAer/` builds a 1.6 M-cell version of the OpenFOAM HPC committee's occDrivAer car case (steady RANS, k-omega
SST, 47 wall patches) from the committee repository. It stores only the build scripts; the case files are downloaded
because they are CC BY-SA 4.0. See [occDrivAer/README.md](occDrivAer/README.md).
