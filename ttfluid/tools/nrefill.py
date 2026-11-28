"""Count the reader's code-window refills per SIMPLE iteration from an image's shared N program.
vm_ncrisc.cpp:585: before every op, if pc < wb or pc + NMAX > wb + NWIN: wb = pc & ~255, read NWIN words from DRAM.
Usage: nrefill.py IMAGE_DIR NWIN[,NWIN...] [its=9,11,10,124,2,2]     (loop trip counts in program order)"""
import sys, numpy as np
N_END, N_SIGC, N_WAITC, N_SIGB, N_WAITB, N_GATHER, N_HALO, N_JMP, N_BRT, N_STREAM, N_CODE, N_CODEAT = range(12)
z = np.load(sys.argv[1] + "/shared.npz", allow_pickle=True)
n = np.asarray(z["n"]).reshape(-1).astype(np.int64)
its = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "9,11,10,124,2,2").split(",")]
ops, pc = {}, 0
while pc < len(n):
    w = int(n[pc]); op = w & 0xff; tgt = None
    if op == N_GATHER: ln = 3 + 2 * ((w >> 8) & 0xf)
    elif op == N_STREAM:
        p = pc + 2
        for _ in range(int(n[pc + 1]) & 0xFFFF):
            p += 2 + 2 * (int(n[p + 1]) & 0xf)
        ln = p - pc
    elif op == N_BRT: ln, tgt = 3, int(n[pc + 2])
    elif op in (N_HALO, N_CODE): ln = 3
    elif op == N_JMP: ln, tgt = 2, int(n[pc + 1])
    elif op == N_CODEAT: ln = 2
    else:
        assert op <= N_WAITB, (pc, op); ln = 1
    ops[pc] = (op, ln, tgt); pc += ln
assert pc == len(n)
NMAX = max(v[1] for v in ops.values())
lens = sorted((v[1] for v in ops.values()), reverse=True)
back = sorted((p, v[0], v[2]) for p, v in ops.items() if v[2] is not None and v[2] <= p)
fwd = sorted((p, v[0], v[2]) for p, v in ops.items() if v[2] is not None and v[2] > p)
print(f"{sys.argv[1]}: n_words {len(n)}, ops {len(ops)}, NMAX {NMAX}, longest ops {lens[:8]}, ops > 256 words: {sum(l > 256 for l in lens)}, > 512: {sum(l > 512 for l in lens)}")
print("backward branches (pc, op, target, body words):", [(p, 'JMP' if o == N_JMP else 'BRT', t, p - t) for p, o, t in back])
print("forward branches:", len(fwd), [(p, 'JMP' if o == N_JMP else 'BRT', t) for p, o, t in fwd][:40])
# loops: innermost-first by back edge; the outermost back edge (largest body) is the SIMPLE loop
loops = [(t, p) for p, o, t in back]
outer = max(loops, key=lambda l: l[1] - l[0]) if loops else None
inner = sorted(l for l in loops if l != outer)
trip = {l: (its[i] if i < len(its) else 1) for i, l in enumerate(inner)}
def run(NWIN, outer_its=3):
    wb, pc, refills, cnt = 0, 0, [], {l: 0 for l in inner}
    per = {"outer": 0, **{l: 0 for l in inner}}
    oit, steps, log = 0, 0, []
    cur_outer = 0
    while True:
        op, ln, tgt = ops[pc]
        if pc < wb or pc + NMAX > wb + NWIN:
            wb = pc & ~255
            reg = next((l for l in inner if l[0] <= pc <= l[1]), "outer")
            if oit == outer_its - 1:
                per[reg] += 1
        steps += 1
        if op == N_END: break
        nxt = pc + ln
        if tgt is not None:
            if tgt <= pc:                       # back edge
                l = (tgt, pc)
                if l == outer:
                    oit += 1
                    if oit >= outer_its: break
                    nxt = tgt
                    for k in cnt: cnt[k] = 0
                else:
                    cnt[l] += 1
                    if cnt[l] < trip[l]: nxt = tgt
                    elif op == N_JMP:           # unconditional back edge: the exit was a forward branch (handled below)
                        nxt = tgt
                    else: cnt[l] = 0
            else:                               # forward: taken only as the exit of a loop on its last trip
                l = next((l for l in inner if l[0] <= pc <= l[1] and tgt > l[1]), None)
                if l is not None and cnt[l] >= trip[l] - 1 and ops[l[1]][0] == N_JMP:
                    nxt = tgt; cnt[l] = 0
        pc = nxt
        if steps > 50_000_000: print("  step cap"); break
    return per
for NW in [int(x) for x in sys.argv[2].split(",")]:
    NWr = max(NW, -(-(NMAX + 255) // 256) * 256)
    per = run(NWr)
    tot = sum(per.values())
    print(f"NWIN {NWr} (asked {NW}): refills per SIMPLE it {tot} = outer body {per['outer']} + loops " +
          ", ".join(f"[{l[0]}..{l[1]}] x{trip[l]}: {per[l]} ({per[l] / trip[l]:.2f}/trip)" for l in inner) +
          f" | bytes per it {tot * NWr * 4 / 1024:.0f} KB")
