"""Protocol simulator of the device VM: decodes the three encoded streams of every core exactly as the kernels in
resident/device/ do and runs them as 3 * P agents with the device's blocking rules (token CBs of finite depth, halo
credits and data counts, the parity-slotted all-to-all reduction). Values are not computed; branch outcomes are
replayed from a MachineEmu run, so every agent follows the real control flow. It finds deadlocks, stream decoding
mistakes and table inconsistencies on the CPU, where a hang costs nothing.

    DevSim(encoded, branches, depth=64).run() -> dict(ops per agent, steps) or raises Deadlock with every agent's state
"""
import numpy as np

from . import encode as E
from .machine import NH


class Deadlock(RuntimeError):
    pass


class _Virt:
    """A paged compute stream seen at virtual pcs: page k of the virtual stream is DRAM page dl[k]."""
    def __init__(self, S, dl, PW):
        self.S, self.dl, self.PW = S, dl, PW

    def __getitem__(self, v):
        return self.S[self.dl[v // self.PW] * self.PW + v % self.PW]


class DevSim:
    def __init__(self, enc, branches, depth=64, ring=4, code_ring=16):
        """code_ring: pages of the paged compute code ring (enc.page): NCRISC pushes DRAM code pages (N_CODE),
        compute runs at a virtual pc whose page k is the k-th delivered page, PACK releases pages behind it."""
        self.e, self.br, self.depth, self.ring, self.cring = enc, branches, depth, ring, code_ring
        assert not enc.page or enc.loop_pages < code_ring, f"the code ring must hold {enc.loop_pages} pages at once"
        m = enc.m
        self.P = P = m.P
        self.tab = enc.tab
        # per core halo plans from the table: plan 0 at tab_halo; with a directory (more plans) plan q > 0 at
        # tab_halo + t[tab_halo - NH - q] and halo slot s on plan t[tab_halo - 1 - s]
        self.recv, self.send, self.plan = [], [], []
        for i in range(P):
            t, h = self.tab[i], int(enc.tab_halo[i])
            nq = len(m.maps.plans) if enc.halo_dir[i] else 1
            rq, sq = [], []
            for q in range(nq):
                p = h + (int(t[h - NH - q]) if q else 0)
                nr = int(t[p]); p += 1
                rl = []
                for _ in range(nr):
                    rl.append(int(t[p])); p += 3 + int(t[p + 2])   # one word per lane (encode.py runs())
                ns = int(t[p])
                rq.append(rl)
                sq.append([int(x) for x in t[p + 1:p + 1 + ns]])
            self.recv.append(rq)
            self.send.append(sq)
            self.plan.append([int(t[h - 1 - s]) & 0xFF if enc.halo_dir[i] else 0 for s in range(NH)])
        for i in range(P):
            for q, rl in enumerate(self.recv[i]):
                for r in rl:
                    assert i in self.send[r][q], f"plan {q}: core {i} sends to {r}, which does not list it as a sender"

    def _rl(self, i, s):
        return self.recv[i][self.plan[i][s]]

    def _sl(self, i, s):
        return self.send[i][self.plan[i][s]]

    def _halo_ready(self, i, mid, slots, data, hcntN):
        for s, rd in zip(slots, self._halo_parts(i, mid, len(slots))):
            if rd:
                assert s != E.NONE16, f"core {i} map {mid} reads a halo but the gather names no slot"
                if data[i, s] < hcntN[i, s] * len(self._sl(i, s)):
                    return False
        return True

    def _halo_parts(self, core, mid, nparts):
        """per part: whether this core reads the halo in it (n_halo > 0)"""
        t = self.tab[core]
        q = int(t[1 + mid]) & (0xFFFFF if self.e.tab_stream is not None else 0xFFFFFFFF)   # streamed: Tm << 20 above
        if self.e.tab_stream is not None:     # streamed: [base, halo part mask, ...]
            return [bool(int(t[q + 1]) >> k & 1) for k in range(nparts)]
        out = []
        for _ in range(nparts):
            nl, nh = int(t[q]), int(t[q + 1])
            q += 2 + nl + nh
            out.append(nh > 0)
        return out

    def run(self, max_steps=10 ** 8):
        P, D = self.P, self.depth
        streams = {"c": self.e.c, "b": self.e.b, "n": self.e.n}
        pcs = self.pcs = {(i, a): 0 for i in range(P) for a in "cbn"}
        bri = {(i, a): 0 for i in range(P) for a in "cbn"}
        done = set()
        self.tloop = {}
        self.dlv = {i: [] for i in range(P)}      # paged code: DRAM pages delivered to core i, in order
        self.rel = {i: 0 for i in range(P)}       # pages compute released
        self.cur = {i: 0 for i in range(P)}       # NCRISC's next page to push
        self.jumped = {i: False for i in range(P)}
        self.seen = {i: 0 for i in range(P)}
        q = {(i, k): 0 for i in range(P) for k in ("cb", "bc", "cn", "nc", "bn", "nb", "rin", "rout")}
        self.prog = {}      # (core, agent) -> pages done in the current N_STREAM / B_SINK
        hcntB = np.zeros((P, NH), int)
        hcntN = np.zeros((P, NH), int)
        credit = np.zeros((P, NH), int)
        data = np.zeros((P, NH), int)
        land = np.zeros((2, P), int)          # land[par][j]: seq core j last published into parity par
        nred = np.zeros(P, int)
        inred = np.zeros(P, bool)
        ops = {a: 0 for a in "cbn"}
        steps = 0
        agents = [(i, a) for i in range(P) for a in "cbn"]
        while len(done) < len(agents):
            progressed = False
            for ag in agents:
                if ag in done:
                    continue
                for _ in range(64):     # run an agent until it blocks (or a slice)
                    r = self._step(ag, streams, pcs, bri, q, hcntB, hcntN, credit, data, land, nred, inred, D)
                    if r is None:
                        break
                    progressed = True
                    ops[ag[1]] += 1
                    steps += 1
                    if r == "end":
                        done.add(ag)
                        break
            if not progressed:
                blocked = {}
                for ag in agents:
                    if ag not in done:
                        S_ = streams[ag[1]]
                        if ag[1] == "c" and self.e.page:
                            dl, v = self.dlv[ag[0]], pcs[ag]
                            S_ = _Virt(S_, dl, E.PW) if v // E.PW < len(dl) else None
                        w = int(S_[pcs[ag]]) if S_ is not None else 0xFF
                        blocked.setdefault((ag[1], w & 0xff), []).append(ag[0])
                raise Deadlock(f"deadlock after {steps} steps; blocked (agent, op): cores "
                               + "; ".join(f"{k}: {v[:8]}{'...' if len(v) > 8 else ''} ({len(v)})"
                                           for k, v in blocked.items()))
            if steps > max_steps:
                raise RuntimeError("step limit")
        return dict(ops=ops, steps=steps, reductions=int(nred[0]), branches=bri[(0, "c")])

    def _branch(self, ag, bri, neg=0):   # neg: the encoder's fused back edge branches on a zero flag
        k = bri[ag]
        bri[ag] += 1
        assert k < len(self.br), f"{ag} executes more branches than the emulator ({len(self.br)})"
        return self.br[k] != bool(neg & 1)

    def _step(self, ag, streams, pcs, bri, q, hcntB, hcntN, credit, data, land, nred, inred, D):
        """Execute one op of agent ag if it can proceed: returns 'ok', 'end' or None (blocked)."""
        i, a = ag
        S = streams[a]
        pc = pcs[ag]
        if not (a == "c" and self.e.page):
            w = int(S[pc])
            op, f0 = (w & 0x7f if a == "c" else w & 0xff), (w >> 8) & 0xf

        def put(k):
            if q[(i, k)] >= D:
                return False
            q[(i, k)] += 1
            return True

        def take(k):
            if q[(i, k)] == 0:
                return False
            q[(i, k)] -= 1
            return True

        if a == "c" and self.e.page:
            PW, dl = E.PW, self.dlv[i]
            g = pc // PW
            if g >= len(dl):
                return None
            if g and pc % PW == 0:
                lp = self.tloop.get(ag)
                self.rel[i] = max(self.rel[i], (lp[2] // PW) if lp and lp[0] < lp[1] else g)
            w = int(S[dl[g] * PW + pc % PW])
            op, f0 = w & 0x7f, (w >> 8) & 0xf
            if op in (E.C_JMP, E.C_BRT):
                if op == E.C_BRT:
                    self._branch(ag, bri, f0)
                pcs[ag] = (pc // PW + 1) * PW
                self.jumped[i] = True
                return "ok"
            if g > self.seen[i]:        # first entry into virtual page g: sequential entry needs the next DRAM page
                if not self.jumped[i] and dl[g] != dl[g - 1] + 1:
                    raise AssertionError(f"core {i}: code page {dl[g]} follows {dl[g - 1]} without a jump")
                self.seen[i] = g
            self.jumped[i] = False
            S = _Virt(S, dl, PW)
        if a == "c":
            if op == E.C_END:
                return "end"
            if op == E.C_FILL:
                pcs[ag] = pc + 2
            elif op in (E.C_LD, E.C_ST, E.C_RVF):
                pcs[ag] = pc + 1
            elif op == E.C_SIGB:
                if not put("cb"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.C_SIGN:
                if not put("cn"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.C_WAITB:
                if not take("bc"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.C_WAITN:
                if not take("nc"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.C_JMP:
                pcs[ag] = int(S[pc + 1])
            elif op == E.C_BRT:
                pcs[ag] = int(S[pc + 2]) if self._branch(ag, bri, f0) else pc + 3
            elif op == E.C_LDS:
                pcs[ag] = pc + 2
            elif op in (E.C_LDR, E.C_LDRH):
                if q[(i, "rin")] == 0:
                    return None
                q[(i, "rin")] -= 1
                pcs[ag] = pc + 1
            elif op in (E.C_STW, E.C_STWH):
                if q[(i, "rout")] >= self.ring:
                    return None
                q[(i, "rout")] += 1
                pcs[ag] = pc + 1
            elif op == E.C_TBEG:
                self.tloop[ag] = [0, w >> 16, pc + 1]
                pcs[ag] = pc + 1
            elif op == E.C_TEND:
                lp = self.tloop[ag]
                lp[0] += 1
                pcs[ag] = lp[2] if lp[0] < lp[1] else pc + 1
            elif op <= E.C_BRT or op == E.C_PH:     # C_PH: a phase marker, no effect
                pcs[ag] = pc + 1
            else:
                raise AssertionError(f"bad compute op {op} at {pc}")
            return "ok"
        if a == "b":
            if op == E.B_END:
                return "end"
            if op == E.B_SIGC:
                if not put("bc"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.B_WAITC:
                if not take("cb"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.B_SIGN:
                if not put("bn"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.B_WAITN:
                if not take("nb"):
                    return None
                pcs[ag] = pc + 1
            elif op == E.B_HALO:
                s = int(S[pc + 1])
                k = hcntB[i, s] + 1
                rl = self._rl(i, s)
                if rl and credit[i, s] < k * len(rl):
                    return None
                hcntB[i, s] = k
                for r in rl:
                    data[r, s] += 1
                pcs[ag] = pc + 3
            elif op == E.B_RED:
                if not inred[i]:
                    nred[i] += 1
                    land[nred[i] & 1, i] = nred[i]
                    inred[i] = True
                par = nred[i] & 1
                if not np.all(land[par] == nred[i]):
                    return None
                inred[i] = False
                pcs[ag] = pc + 3 + 2 * ((w >> 8) & 0xf)   # red2: two or three (part, fold) pairs
            elif op == E.B_OUT:
                pcs[ag] = pc + 3
            elif op == E.B_SINK:
                n, tn = int(S[pc + 1]) & 0xFFFF, int(S[pc + 1]) >> 16 or self.e.m.T
                k = self.prog.get(ag, 0)
                if k < n * tn:
                    if q[(i, "rout")] == 0:
                        return None
                    q[(i, "rout")] -= 1
                    self.prog[ag] = k + 1
                    return "ok"
                self.prog[ag] = 0
                pcs[ag] = pc + 2 + n
            elif op == E.B_FETCH:
                pcs[ag] = pc + 4
            elif op == E.B_SCAL:
                pcs[ag] = pc + 2 + int(S[pc + 1])
            elif op == E.B_GATHER:      # bgather: dst, map, one L1 source per part, no halo
                pcs[ag] = pc + 3 + ((w >> 8) & 0xf)
            elif op == E.B_JMP:
                pcs[ag] = int(S[pc + 1])
            elif op == E.B_BRT:
                pcs[ag] = int(S[pc + 2]) if self._branch(ag, bri, f0) else pc + 3
            elif op == E.B_PH:      # a phase marker, no effect
                pcs[ag] = pc + 1
            else:
                raise AssertionError(f"bad brisc op {op} at {pc}")
            return "ok"
        if op == E.N_END:
            return "end"
        if op == E.N_SIGC:
            if not put("nc"):
                return None
            pcs[ag] = pc + 1
        elif op == E.N_WAITC:
            if not take("cn"):
                return None
            pcs[ag] = pc + 1
        elif op == E.N_SIGB:
            if not put("nb"):
                return None
            pcs[ag] = pc + 1
        elif op == E.N_WAITB:
            if not take("bn"):
                return None
            pcs[ag] = pc + 1
        elif op == E.N_GATHER:
            mid = int(S[pc + 2])
            slots = [int(x) for x in S[pc + 4:pc + 3 + 2 * f0:2]]
            if (int(S[pc]) >> 12) & 0xf != 1 and not self._halo_ready(i, mid, slots, data, hcntN):
                return None
            gt = (int(S[pc]) >> 16) & 0xff     # gtile: gt - 1 tokens to compute, here all at once (stricter)
            if gt > 1:
                if q[(i, "nc")] + gt - 1 > D:
                    return None
                q[(i, "nc")] += gt - 1
            pcs[ag] = pc + 3 + 2 * f0
        elif op == E.N_STREAM:
            n, tn = int(S[pc + 1]) & 0xFFFF, int(S[pc + 1]) >> 16 or self.e.m.T
            k = self.prog.get(ag, 0)
            ents, p = [], pc + 2
            for _ in range(n):
                x = int(S[p + 1])
                mid1, npart = (x >> 4) & (0x3ff if E.WIDE else 0xfff), x & 0xf
                ents.append((mid1, [int(y) for y in S[p + 3:p + 2 + 2 * npart:2]]))
                p += 2 + 2 * npart
            if k == 0:
                for mid1, slots in ents:
                    if mid1 and not self._halo_ready(i, mid1 - 1, slots, data, hcntN):
                        return None
            if k < n * tn:
                if q[(i, "rin")] >= self.ring:
                    return None
                q[(i, "rin")] += 1
                self.prog[ag] = k + 1
                return "ok"
            self.prog[ag] = 0
            pcs[ag] = p
        elif op == E.N_HALO:
            s = int(S[pc + 1])
            hcntN[i, s] += 1
            for j in self._sl(i, s):
                credit[j, s] += 1
            pcs[ag] = pc + 3
        elif op == E.N_CODE:
            hi, want = int(S[pc + 1]), int(S[pc + 2])
            free = len(self.dlv[i]) - self.rel[i] < self.cring
            if self.cur[i] < hi or self.cur[i] < want and free:
                if not free:
                    return None
                self.dlv[i].append(self.cur[i])
                self.cur[i] += 1
                return "ok"
            pcs[ag] = pc + 3
        elif op == E.N_CODEAT:
            self.cur[i] = int(S[pc + 1])
            pcs[ag] = pc + 2
        elif op == E.N_JMP:
            pcs[ag] = int(S[pc + 1])
        elif op == E.N_BRT:
            pcs[ag] = int(S[pc + 2]) if self._branch(ag, bri, f0) else pc + 3
        else:
            raise AssertionError(f"bad ncrisc op {op} at {pc}")
        return "ok"
