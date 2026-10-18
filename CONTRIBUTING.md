# Contributing to tt-fluid

Thanks for taking the time. Bug reports, cases that fail, new boundary conditions and performance work are all welcome.

## Set up

```bash
git clone https://github.com/moritztng/tt-fluid.git && cd tt-fluid
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'            # add ,tenstorrent for device runs
ruff check . && pytest -q          # what CI runs, ~20 s
```

## The rules every change keeps

1. **The solver stays OpenFOAM's.** A change to `ttfluid/solver` keeps the tests in `tests/test_vs_openfoam.py` green.
   A new feature (a boundary condition, a scheme) comes with a reference from OpenFOAM v2412 on a small case.
2. **The device matches the emulator bit for bit.** A change to the lowering, the encoding or the kernels keeps
   `tests/test_emulator.py` green, and a device run of the changed program reports `vs_emu.worst_rel` 0.
3. **A performance change says what it does to the answer.** Either the final state is bitwise unchanged (say so and
   how you checked), or the change is graded like any other: windowed Cd/Cl against OpenFOAM, `bench/nvidia/cdwindow.py`.
4. **Every number has its data.** A claim in the docs points at a script and the raw files it reads, with the chip clock
   recorded during the run.

## Pull requests

- Keep a pull request to one change, with a description of what it does and how it was tested (CPU tests, and the
  device and clock if it touches the device path).
- Run `ruff check .` and `pytest -q` before pushing.
- Device runs: always bound them (`timeout -s TERM ...`) and never kill a device process with SIGKILL; a killed process
  can leave the card needing a reset.
- By contributing you agree that your contribution is licensed under GPL-3.0-or-later.

## Reporting a bug

Open an issue with the command, the full log, `pip list | grep -E 'ttnn|numpy|scipy|tt-fluid'`, the card (`tt-smi`) and,
if you can share it, the case (or a smaller case that shows the same thing).
