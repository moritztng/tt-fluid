**What this changes**

**How it was tested**
- [ ] `ruff check .` and `pytest -q`
- [ ] device run (card, chips, AICLK during the run), `vs_emu.worst_rel` = 0, if it touches the device path
- [ ] final state bitwise unchanged, or graded against OpenFOAM, if it is a performance change

See [CONTRIBUTING.md](../CONTRIBUTING.md).
