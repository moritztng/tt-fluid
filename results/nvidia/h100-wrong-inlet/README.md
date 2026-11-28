Two SPUMA SPDP runs of HPC_motorbike Small that are wrong on purpose, to show that `bench/cdgrade.py` can fail a run.
The inlet velocity in `0/U` is 20.2 m/s (`x1.010.dat`) and 20.06 m/s (`x1.003.dat`) instead of 20, while the force
coefficients are still normalised with 20 m/s. Everything else is the setup of `../dgx-h100`: one H100 SXM, 800
iterations from the potentialFoam start, 2026-11-28, both cases on the same GPU under MPS.

The grader reads +82.0 and +43.5 counts on iterations 201-800 against a band of 17.8 and fails both
(`tests/test_cdgrade.py`, docs/accuracy.md section 3). A third run with one GAMG cycle per SIMPLE iteration for the
pressure diverged at iteration 16 and left no window to grade.
