"""Blackhole Tensix instruction words for generated SFPU bodies (LEVERS 13; the inline program of C_BLK), and ROWS,
the per-row instruction sequence of every stock tile op a body may replace.

Encoders: one function per instruction, named after it in lower case, arguments in the order of the LLK's TT_OP_
macro (tt_llk_blackhole/common/inc/ckernel_ops.h); LAYOUT lists them. The result is the word kernel code pushes to
the instruction buffer, TT_OP(opcode, params) = opcode << 24 | params. Inline in a kernel ELF the same instruction
is that word rotated left by two bits (inline()): that is what objdump prints, and selftest() checks every encoder
against the words of the compiled VM kernel (asm() reads objdump's operand syntax).

    python -m ttfluid.resident.tensix                 the vectors kept below
    python -m ttfluid.resident.tensix FILE.dis        every Tensix line of an objdump -d of a TRISC kernel
"""
import re

# ---- registers and modes (ckernel_instr_params.h p_sfpu, sfpi_constants.h) ----
LREG0, LREG1, LREG2, LREG3, LREG4, LREG5, LREG6, LREG7 = range(8)
LCONST_0_8373, LCONST_0, LCONST_1 = 8, 9, 10      # fixed constants 0.8373, 0.0, 1.0
LREG11, LREG12, LREG13, LREG14 = 11, 12, 13, 14   # programmable constants (SFPCONFIG); init_sfpu sets L11 = -1.0
LCONST_neg1 = 11
LTILEID = 15
FP32 = 3                                          # SFPLOAD / SFPSTORE mod0 (InstrModLoadStore::FP32); 0 = as Dest's format
MOD0_DEFAULT = 0
ADDR_MOD_6, ADDR_MOD_7 = 6, 7                     # 3-bit field; blk_init: 7 = no step, 6 = dest += 2
SFPLOADI_MOD0_FLOATB, SFPLOADI_MOD0_FLOATA, SFPLOADI_MOD0_USHORT, SFPLOADI_MOD0_SHORT = 0, 1, 2, 4
SFPLOADI_MOD0_UPPER, SFPLOADI_MOD0_LOWER = 8, 10  # high / low 16 bits of the LREG, the other half kept
SFPMAD_MOD1_NEGATE_VA, SFPMAD_MOD1_NEGATE_VC, SFPMAD_MOD1_INDIRECT_VA, SFPMAD_MOD1_INDIRECT_VD = 1, 2, 4, 8
SFPMOV_MOD1_COPY, SFPMOV_MOD1_COMPSIGN, SFPMOV_MOD1_ALL_LANES, SFPMOV_MOD1_CONFIG = 0, 1, 2, 8
SFPABS_MOD1_INT, SFPABS_MOD1_FLOAT = 0, 1
SFPSETCC_MOD1_LREG_LT0, SFPSETCC_MOD1_IMM_BIT0, SFPSETCC_MOD1_LREG_NE0 = 0, 1, 2
SFPSETCC_MOD1_LREG_GTE0, SFPSETCC_MOD1_LREG_EQ0, SFPSETCC_MOD1_COMP = 4, 6, 8
SFPENCC_IMM12_NEITHER, SFPENCC_IMM12_BOTH = 0, 3
SFPENCC_MOD1_EU_R1, SFPENCC_MOD1_EC_R1, SFPENCC_MOD1_EI_R1 = 0, 1, 2
SFPENCC_MOD1_EU_RI, SFPENCC_MOD1_EC_RI, SFPENCC_MOD1_EI_RI = 8, 9, 10
SFPPUSHC_MOD1_PUSH, SFPPUSHC_MOD1_REPLACE = 0, 1
SFPSWAP_MOD1_SWAP, SFPSWAP_MOD1_VEC_MIN_MAX, SFPSWAP_MOD1_VEC_MAX_MIN = 0, 1, 9
SFPSETSGN_MOD1_ARG_IMM = 1
# SFPCONFIG destinations: load-macro instruction templates 0-3, sequences 4-7, Misc 8, constants 11-14, control 15
SFPCONFIG_DEST_MACRO_INST0, SFPCONFIG_DEST_MACRO_SEQ0, SFPCONFIG_DEST_MACRO_CTRL = 0, 4, 8
SFPCONFIG_DEST_LREG11, SFPCONFIG_DEST_SFPU_CTRL = 11, 15
SETRWC_CR_D, SETRWC_SET_D, SETRWC_SET_ABD_F = 4, 4, 15    # p_setrwc: rwc_cr CR_D; BitMask SET_D, SET_ABD_F
STALL_SFPU, STALL_CFG, WAIT_MATH, WAIT_SFPU = 0x100, 0x80, 0x10, 0x800   # p_stall (stallwait(stall_res, wait_res))

_I12 = (("imm12_math", 12, 12), ("lreg_c", 8, 4), ("lreg_dest", 4, 4), ("instr_mod1", 0, 4))
_MAD = (("lreg_src_a", 16, 4), ("lreg_src_b", 12, 4), ("lreg_src_c", 8, 4), ("lreg_dest", 4, 4), ("instr_mod1", 0, 4))
_LS = (("lreg_ind", 20, 4), ("instr_mod0", 16, 4), ("sfpu_addr_mode", 13, 3), ("dest_reg_addr", 0, 10))
_I16 = (("imm16_math", 8, 16), ("lreg_dest", 4, 4), ("instr_mod1", 0, 4))
# name: (opcode, ((field, shift, bits), ...) in the TT_OP_ macro's argument order)
LAYOUT = {
    "sfpload": (0x70, _LS), "sfpstore": (0x72, _LS), "sfploadmacro": (0x93, _LS),
    "sfploadi": (0x71, (("lreg_ind", 20, 4), ("instr_mod0", 16, 4), ("imm16", 0, 16))),
    "sfpmad": (0x84, _MAD), "sfpadd": (0x85, _MAD), "sfpmul": (0x86, _MAD), "sfpmul24": (0x98, _MAD),
    "sfpaddi": (0x75, _I16), "sfpmuli": (0x74, _I16),
    "sfpnop": (0x8f, ()),
    "sfpsetcc": (0x7b, _I12), "sfpencc": (0x8a, _I12), "sfppushc": (0x87, _I12), "sfppopc": (0x88, _I12),
    "sfpcompc": (0x8b, _I12), "sfpmov": (0x7c, _I12), "sfpabs": (0x7d, _I12), "sfpsetsgn": (0x89, _I12),
    "sfpswap": (0x92, _I12), "sfpgt": (0x97, _I12), "sfple": (0x96, _I12), "sfplz": (0x81, _I12),
    "sfparecip": (0x99, _I12), "sfpiadd": (0x79, _I12), "sfpdivp2": (0x76, _I12), "sfpexexp": (0x77, _I12),
    "sfpexman": (0x78, _I12), "sfpsetexp": (0x82, _I12), "sfpsetman": (0x83, _I12), "sfpand": (0x7e, _I12),
    "sfpor": (0x7f, _I12), "sfpxor": (0x8d, _I12), "sfpnot": (0x80, _I12), "sfpshft": (0x7a, _I12),
    "sfpshft2": (0x94, _I12), "sfptransp": (0x8c, _I12),
    "sfpcast": (0x90, (("lreg_src_c", 8, 4), ("lreg_dest", 4, 4), ("instr_mod1", 0, 4))),
    "sfpconfig": (0x91, (("imm16_math", 8, 16), ("config_dest", 4, 4), ("instr_mod1", 0, 4))),
    "replay": (0x04, (("start_idx", 14, 10), ("len", 4, 10), ("execute_while_loading", 1, 1), ("load_mode", 0, 1))),
    "incrwc": (0x38, (("rwc_cr", 18, 6), ("rwc_d", 14, 4), ("rwc_b", 10, 4), ("rwc_a", 6, 4))),
    "setrwc": (0x37, (("clear_ab_vld", 22, 2), ("rwc_cr", 18, 4), ("rwc_d", 14, 4), ("rwc_b", 10, 4), ("rwc_a", 6, 4),
                      ("BitMask", 0, 6))),
    "nop": (0x02, ()),
    "stallwait": (0xa2, (("stall_res", 15, 9), ("wait_res", 0, 15))),
    "setc16": (0xb2, (("setc16_reg", 16, 8), ("setc16_value", 0, 16))),
}
_IMM = {"imm12_math", "imm16_math", "imm16", "setc16_value"}     # immediates may be given signed
_BYOP = {op: name for name, (op, _) in LAYOUT.items()}


def _mk(name):
    op, fields = LAYOUT[name]

    def enc(*args):
        assert len(args) == len(fields), "%s(%s)" % (name, ", ".join(f for f, _, _ in fields))
        w = op << 24
        for (f, sh, bits), v in zip(fields, args):
            v = int(v)
            if f in _IMM:
                assert -(1 << bits - 1) <= v < 1 << bits, (name, f, v)
                v &= (1 << bits) - 1
            assert 0 <= v < 1 << bits, (name, f, v)
            w |= v << sh
        return w
    enc.__name__ = name
    enc.__doc__ = "%s(%s) -> instruction word" % (name, ", ".join(f for f, _, _ in fields))
    return enc


for _n in LAYOUT:
    globals()[_n] = _mk(_n)


def inline(w):
    """The word as it stands inline in kernel code (and in objdump -d): rotated left by two bits."""
    return ((w << 2) | (w >> 30)) & 0xFFFFFFFF


def from_inline(raw):
    return ((raw >> 2) | (raw << 30)) & 0xFFFFFFFF


def dis(w):
    """'name field=value ...' of an instruction word (fields the layout does not cover show as rest=)."""
    name = _BYOP.get(w >> 24)
    if name is None:
        return "op_%02x 0x%06x" % (w >> 24, w & 0xFFFFFF)
    out, rest = [name], w & 0xFFFFFF
    for f, sh, bits in LAYOUT[name][1]:
        v = (w >> sh) & ((1 << bits) - 1)
        rest &= ~(((1 << bits) - 1) << sh)
        out.append("%s=%d" % (f, v))
    return " ".join(out) + (" rest=0x%x" % rest if rest else "")


def loadmacro(macro, vd, instr_mod0, sfpu_addr_mode, slot_addr):
    """SFPLOADMACRO of sequence `macro` (0-3) loading Dest address slot_addr (even) into LReg vd (0-7). The LReg's
    high bit travels in the low bit of the address field: VD = (Addr & 1) << 2 | VDLo with Addr the full Dest address
    (immediate + offsets + row counter), so the row counter and offsets must be even, as they are in fp32 bodies."""
    assert 0 <= macro < 4 and 0 <= vd < 8 and slot_addr % 2 == 0
    return sfploadmacro(macro << 2 | (vd & 3), instr_mod0, sfpu_addr_mode, slot_addr | vd >> 2)


def seq_bits(insn, delay=0, use_vb=False, to_l16=False):
    """One byte of a load-macro sequence: insn 0 = nothing, 2 = SFPNOP, 3 = SFPSTORE, 4-7 = instruction template 0-3;
    delay 0-7 (executes delay + 1 cycles after the SFPLOADMACRO); use_vb (0x80): the loaded LReg replaces the
    template's VB (else VC), for the store unit: store LREG0; to_l16 (0x40): the result goes to LReg[16], for the
    store unit: LReg[16] is what is stored."""
    assert insn in (0, 2, 3, 4, 5, 6, 7) and 0 <= delay < 8
    return (0x80 if use_vb else 0) | (0x40 if to_l16 else 0) | delay << 3 | insn


def macro_sequence(m, simple=0, mad=0, rnd=0, store=0):
    """The three instructions that write load-macro sequence m (0-3) in full: bytes for the simple, MAD, round and
    store units (seq_bits). Uses LREG0 as scratch."""
    return [sfploadi(LREG0, SFPLOADI_MOD0_LOWER, mad << 8 | simple), sfploadi(LREG0, SFPLOADI_MOD0_UPPER, store << 8 | rnd),
            sfpconfig(0, SFPCONFIG_DEST_MACRO_SEQ0 + m, 0)]


def macro_misc(store_mod0=0, uses_load_mod0=0, unit_delay_kind=0):
    """SFPCONFIG of the load-macro Misc word: StoreMod0 (4 b), UsesLoadMod0ForStore (1 b per sequence),
    UnitDelayKind (1 b per sub-unit)."""
    return sfpconfig(unit_delay_kind << 8 | uses_load_mod0 << 4 | store_mod0, SFPCONFIG_DEST_MACRO_CTRL, 1)


def template(k, word):
    """An instruction word turned into the write of load-macro instruction template k (0-3): the same instruction
    with VD = 12 + k (no LReg is written)."""
    name = _BYOP[word >> 24]
    f = dict((x[0], x) for x in LAYOUT[name][1])
    _, sh, bits = f["lreg_dest"]
    return (word & ~(((1 << bits) - 1) << sh)) | (12 + k) << sh


def fconst(lreg, bits):
    """The two SFPLOADI that set LReg to the fp32 bit pattern `bits` (upper half first, as the compiler emits)."""
    return [sfploadi(lreg, SFPLOADI_MOD0_UPPER, bits >> 16), sfploadi(lreg, SFPLOADI_MOD0_LOWER, bits & 0xFFFF)]


# ---- objdump syntax ----
_L = lambda s: int(s[1:]) if s.startswith("L") else int(s, 0)
# mnemonic: (encoder, operand fields in printed order, fields objdump leaves out)
SYNTAX = {
    "sfpload": ("sfpload", ("lreg_ind", "dest_reg_addr", "instr_mod0", "sfpu_addr_mode"), {}),
    "sfpstore": ("sfpstore", ("lreg_ind", "dest_reg_addr", "instr_mod0", "sfpu_addr_mode"), {}),
    "sfploadi": ("sfploadi", ("lreg_ind", "imm16", "instr_mod0"), {}),
    "sfpmad": ("sfpmad", ("lreg_dest", "lreg_src_a", "lreg_src_b", "lreg_src_c", "instr_mod1"), {}),
    "sfpadd": ("sfpadd", ("lreg_dest", "lreg_src_b", "lreg_src_c", "instr_mod1"), {"lreg_src_a": LCONST_1}),
    "sfpmul": ("sfpmul", ("lreg_dest", "lreg_src_a", "lreg_src_b", "instr_mod1"), {"lreg_src_c": LCONST_0}),
    "sfpaddi": ("sfpaddi", ("lreg_dest", "imm16_math", "instr_mod1"), {}),
    "sfpmuli": ("sfpmuli", ("lreg_dest", "imm16_math", "instr_mod1"), {}),
    "sfpnop": ("sfpnop", (), {}),
    "sfpsetcc": ("sfpsetcc", ("lreg_c", "imm12_math", "instr_mod1"), {"lreg_dest": 0}),
    "sfpencc": ("sfpencc", ("imm12_math", "instr_mod1"), {"lreg_c": 0, "lreg_dest": 0}),
    "sfppushc": ("sfppushc", ("instr_mod1",), {"imm12_math": 0, "lreg_c": 0, "lreg_dest": 0}),
    "sfppopc": ("sfppopc", ("instr_mod1",), {"imm12_math": 0, "lreg_c": 0, "lreg_dest": 0}),
    "sfpcompc": ("sfpcompc", (), {"imm12_math": 0, "lreg_c": 0, "lreg_dest": 0, "instr_mod1": 0}),
    "sfptransp": ("sfptransp", (), {"imm12_math": 0, "lreg_c": 0, "lreg_dest": 0, "instr_mod1": 0}),
    "sfpmov": ("sfpmov", ("lreg_dest", "lreg_c", "instr_mod1"), {"imm12_math": 0}),
    "sfpabs": ("sfpabs", ("lreg_dest", "lreg_c", "instr_mod1"), {"imm12_math": 0}),
    "sfpswap": ("sfpswap", ("lreg_dest", "lreg_c", "instr_mod1"), {"imm12_math": 0}),
    "sfparecip": ("sfparecip", ("lreg_dest", "lreg_c", "instr_mod1"), {"imm12_math": 0}),
    "sfpexexp": ("sfpexexp", ("lreg_dest", "lreg_c", "instr_mod1"), {"imm12_math": 0}),
    "sfpnot": ("sfpnot", ("lreg_dest", "lreg_c"), {"imm12_math": 0, "instr_mod1": 0}),
    "sfpcast": ("sfpcast", ("lreg_dest", "lreg_src_c", "instr_mod1"), {}),
    "sfpsetsgn": ("sfpsetsgn", ("lreg_dest", "lreg_c", "imm12_math", "instr_mod1"), {}),
    "sfpiadd": ("sfpiadd", ("lreg_dest", "lreg_c", "imm12_math", "instr_mod1"), {}),
    "sfpsetexp": ("sfpsetexp", ("lreg_dest", "lreg_c", "imm12_math", "instr_mod1"), {}),
    "sfpshft": ("sfpshft", ("lreg_dest", "lreg_c", "imm12_math", "instr_mod1"), {}),
    "sfpshft2": ("sfpshft2", ("lreg_dest", "lreg_c", "imm12_math", "instr_mod1"), {}),
    "sfpconfig": ("sfpconfig", ("config_dest", "imm16_math", "instr_mod1"), {}),
    "ttreplay": ("replay", ("start_idx", "len", "execute_while_loading", "load_mode"), {}),
    "ttincrwc": ("incrwc", ("rwc_cr", "rwc_d", "rwc_b", "rwc_a"), {}),
    "ttsetrwc": ("setrwc", ("clear_ab_vld", "rwc_cr", "rwc_d", "rwc_b", "rwc_a", "BitMask"), {}),
    "ttstallwait": ("stallwait", ("stall_res", "wait_res"), {}),
    "ttsetc16": ("setc16", ("setc16_reg", "setc16_value"), {}),
    "ttnop": ("nop", (), {}),
}


def asm(text):
    """The word of one instruction in objdump's syntax, e.g. asm("sfpmad L5,L2,L4,L0,0")."""
    mn, _, ops = text.strip().partition(" ")
    mn, ops = mn.strip(), [x for x in re.split(r"[,\s]+", ops.strip()) if x]
    name, order, implied = SYNTAX[mn]
    assert len(ops) == len(order), (text, order)
    val = dict(implied)
    val.update(zip(order, map(_L, ops)))
    return globals()[name](*[val[f] for f, _, _ in LAYOUT[name][1]])


# (objdump text, inline word) from the VM compute kernel as compiled for TRISC1 (vm_compute.cpp, anx-glx JIT cache)
VECTORS = [
    ("sfpabs L0,L0,1", 0xf4000005),
    ("sfpabs L2,L1,1", 0xf4000485),
    ("sfpadd L0,L0,L1,0", 0x14280402),
    ("sfpadd L0,L0,L1,2", 0x1428040a),
    ("sfpadd L3,L0,L10,2", 0x142828ca),
    ("sfpadd L0,L0,L10,0", 0x14282802),
    ("sfpadd L4,L14,L1,0", 0x142b8502),
    ("sfpadd L1,L0,L3,2", 0x14280c4a),
    ("sfpadd L2,L10,L2,0", 0x142a8882),
    ("sfpadd L4,L0,L10,2", 0x1428290a),
    ("sfpaddi L0,19264,0", 0xd52d0001),
    ("sfpaddi L0,16128,0", 0xd4fc0001),
    ("sfparecip L4,L0,0", 0x64000102),
    ("sfparecip L0,L1,0", 0x64000402),
    ("sfparecip L0,L2,0", 0x64000802),
    ("sfpcast L1,L1,0", 0x40000442),
    ("sfpcast L0,L1,0", 0x40000402),
    ("sfpcompc", 0x2c000002),
    ("sfpconfig 15,0,1", 0x440003c6),
    ("sfpconfig 12,0,0", 0x44000302),
    ("sfpconfig 4,0,0", 0x44000102),
    ("sfpconfig 5,0,0", 0x44000142),
    ("sfpconfig 8,816,1", 0x440cc206),
    ("sfpconfig 13,0,0", 0x44000342),
    ("sfpconfig 14,0,0", 0x44000382),
    ("sfpconfig 5,4,1", 0x44001146),
    ("sfpencc 0x003,10", 0x2800c02a),
    ("sfpexexp L1,L0,0", 0xdc000041),
    ("sfpexexp L3,L0,0", 0xdc0000c1),
    ("sfpexexp L0,L1,1", 0xdc000405),
    ("sfpiadd L0,L0,0x001,5", 0xe4004015),
    ("sfpiadd L2,L0,0x000,6", 0xe4000099),
    ("sfpiadd L3,L1,0xF80,5", 0xe7e004d5),
    ("sfpiadd L1,L1,0x001,5", 0xe4004455),
    ("sfpiadd L0,L12,0x000,6", 0xe4003019),
    ("sfpiadd L3,L2,0x000,2", 0xe40008c9),
    ("sfpiadd L1,L3,0xF01,5", 0xe7c04c55),
    ("sfpiadd L3,L0,0x000,6", 0xe40000d9),
    ("sfpload L2,0,0,7", 0xc0838001),
    ("sfpload L0,0,0,7", 0xc0038001),
    ("sfpload L1,0,0,7", 0xc0438001),
    ("sfpload L4,0,0,7", 0xc1038001),
    ("sfploadi L5,-21845,10", 0xc56aaaad),
    ("sfploadi L0,15802,8", 0xc420f6e9),
    ("sfploadi L0,11916,10", 0xc428ba31),
    ("sfploadi L5,15843,8", 0xc560f78d),
    ("sfploadi L5,-29127,10", 0xc56a38e5),
    ("sfploadi L5,15890,8", 0xc560f849),
    ("sfploadi L5,18725,10", 0xc5692495),
    ("sfploadi L5,15948,8", 0xc560f931),
    ("sfpmad L0,L4,L0,L5,0", 0x10101402),
    ("sfpmad L5,L2,L4,L0,0", 0x10090142),
    ("sfpmad L5,L0,L4,L12,2", 0x1001314a),
    ("sfpmad L6,L4,L5,L9,3", 0x1011658e),
    ("sfpmad L5,L0,L6,L12,1", 0x1001b146),
    ("sfpmad L4,L6,L5,L9,2", 0x1019650a),
    ("sfpmad L4,L4,L0,L10,0", 0x10102902),
    ("sfpmad L0,L1,L0,L5,0", 0x10041402),
    ("sfpmov L2,L0,0", 0xf0000081),
    ("sfpmov L2,L1,0", 0xf0000481),
    ("sfpmov L1,L0,0", 0xf0000041),
    ("sfpmov L5,L4,0", 0xf0001141),
    ("sfpmov L0,L0,1", 0xf0000005),
    ("sfpmov L2,L9,2", 0xf0002489),
    ("sfpmov L2,L9,0", 0xf0002481),
    ("sfpmov L3,L2,2", 0xf00008c9),
    ("sfpmul L3,L3,L4,0", 0x180d24c2),
    ("sfpmul L4,L1,L3,0", 0x1804e502),
    ("sfpmul L0,L0,L1,0", 0x18006402),
    ("sfpmul L4,L3,L3,0", 0x180ce502),
    ("sfpmul L1,L2,L0,0", 0x18082442),
    ("sfpmul L1,L0,L1,1", 0x18006446),
    ("sfpmul L4,L2,L0,0", 0x18082502),
    ("sfpmul L1,L4,L1,0", 0x18106442),
    ("sfpmuli L3,16384,0", 0xd10000c1),
    ("sfpmuli L0,16128,0", 0xd0fc0001),
    ("sfpmuli L1,16384,0", 0xd1000041),
    ("sfpmuli L0,16384,0", 0xd1000001),
    ("sfpnop", 0x3c000002),
    ("sfpnot L0,L1", 0x00000402),
    ("sfppopc 0", 0x20000002),
    ("sfppushc 0", 0x1c000002),
    ("sfpsetcc L2,0x000,4", 0xec000811),
    ("sfpsetcc L2,0x000,6", 0xec000819),
    ("sfpsetcc L1,0x000,0", 0xec000401),
    ("sfpsetcc L2,0x000,2", 0xec000809),
    ("sfpsetcc L5,0x000,0", 0xec001401),
    ("sfpsetcc L3,0x000,2", 0xec000c09),
    ("sfpsetcc L0,0x000,4", 0xec000011),
    ("sfpsetcc L0,0x000,6", 0xec000019),
    ("sfpsetexp L0,L0,0x07F,1", 0x081fc006),
    ("sfpsetexp L0,L1,0x000,0", 0x08000402),
    ("sfpsetsgn L1,L0,0x001,1", 0x24004046),
    ("sfpsetsgn L3,L0,0x000,0", 0x240000c2),
    ("sfpshft L0,L2,0xFFF,5", 0xebffc815),
    ("sfpshft2 L1,L1,0x000,4", 0x50000452),
    ("sfpshft2 L5,L5,0x000,4", 0x50001552),
    ("sfpshft2 L0,L0,0x000,3", 0x5000000e),
    ("sfpshft2 L4,L4,0x000,3", 0x5000110e),
    ("sfpshft2 L13,L0,0x000,6", 0x5000035a),
    ("sfpstore L2,0,0,7", 0xc8838001),
    ("sfpstore L0,0,0,7", 0xc8038001),
    ("sfpstore L0,0,3,7", 0xc80f8001),
    ("sfpstore L0,2,3,7", 0xc80f8009),
    ("sfpstore L0,16,3,7", 0xc80f8041),
    ("sfpstore L0,18,3,7", 0xc80f8049),
    ("sfpstore L4,0,0,7", 0xc9038001),
    ("sfpswap L12,L2,1", 0x48000b06),
    ("sfpswap L12,L2,9", 0x48000b26),
    ("sfptransp", 0x30000002),
    ("ttincrwc 0,2,0,0", 0xe0020000),
    ("ttreplay 0,6,0,0", 0x10000180),
    ("ttreplay 0,4,0,0", 0x10000100),
    ("ttreplay 0,3,0,0", 0x100000c0),
    ("ttreplay 0,31,0,0", 0x100007c0),
    ("ttreplay 0,4,0,1", 0x10000104),
    ("ttreplay 0,3,0,1", 0x100000c4),
    ("ttreplay 0,6,1,1", 0x1000018c),
    ("ttreplay 0,8,0,0", 0x10000200),
    ("ttsetc16 19,0", 0xc84c0002),
    ("ttsetc16 35,0", 0xc88c0002),
    ("ttsetc16 54,0", 0xc8d80002),
    ("ttsetc16 15,0", 0xc83c0002),
    ("ttsetc16 31,0", 0xc87c0002),
    ("ttsetc16 50,0", 0xc8c80002),
    ("ttsetc16 12,1", 0xc8300006),
    ("ttsetc16 28,1", 0xc8700006),
    ("ttsetrwc 0,4,8,0,0,4", 0xdc480010),
    ("ttsetrwc 0,0,0,0,0,15", 0xdc00003c),
    ("ttsetrwc 0,0,0,0,0,4", 0xdc000010),
    ("ttstallwait 256,16", 0x8a000042),
    ("ttstallwait 2,2064", 0x88042042),
    ("ttstallwait 128,2048", 0x89002002),
    ("ttstallwait 128,16", 0x89000042),
    ("ttstallwait 128,2064", 0x89002042),
]


def selftest(path=None, verbose=False):
    """Every vector (and, with a path, every Tensix line of that objdump -d): asm(text) == the inline word, and
    dis() of it names the same instruction. Returns (checked, mnemonics); raises on the first mismatch."""
    vec = list(VECTORS)
    if path:
        for ln in open(path):
            m = re.match(r"\s+[0-9a-f]+:\s+([0-9a-f]{8})\s+(\S+)\s*(.*)$", ln)
            if m and m.group(2) in SYNTAX:
                vec.append(((m.group(2) + " " + m.group(3)).strip(), int(m.group(1), 16)))
    seen = set()
    for text, raw in vec:
        w = asm(text)
        assert inline(w) == raw and from_inline(raw) == w, "%s: encoded %08x (inline %08x), kernel has %08x" % (text, w, inline(w), raw)
        name = SYNTAX[text.split()[0]][0]
        assert dis(w).split()[0] == name and "rest=" not in dis(w), (text, dis(w))
        seen.add(name)
        if verbose:
            print("%-34s %08x  %s" % (text, w, dis(w)))
    # layouts without a compiled sample: the TT_OP_ macro's arithmetic, by hand
    assert sfploadmacro(5, FP32, ADDR_MOD_6, 64) == 0x93 << 24 | 5 << 20 | 3 << 16 | 6 << 13 | 64
    assert sfpgt(0, 1, 2, 3) == 0x97 << 24 | 1 << 8 | 2 << 4 | 3 and sfple(0, 1, 2, 3) == 0x96 << 24 | 1 << 8 | 2 << 4 | 3
    assert nop() == 0x02000000 and sfpnop() == 0x8F000000
    assert loadmacro(1, 5, FP32, ADDR_MOD_6, 128) == sfploadmacro(1 << 2 | 1, FP32, ADDR_MOD_6, 129)
    assert template(0, sfpadd(LCONST_1, LREG2, 0, 0, 0)) == sfpadd(LCONST_1, LREG2, 0, 12, 0)
    return len(vec), sorted(seen)


if __name__ == "__main__":
    import sys
    n, names = selftest(sys.argv[1] if len(sys.argv) > 1 else None, verbose="-v" in sys.argv)
    print("tensix selftest: %d instruction words match the compiled kernel (%d encoders: %s); not in any compiled "
          "sample: %s" % (n, len(names), " ".join(names), " ".join(sorted(set(LAYOUT) - set(names)))))
