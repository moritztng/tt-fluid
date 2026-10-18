"""Trace the ELL solver into a vector program.

The device holds every value as a 1-D vector over a *space*: the cells, the faces of one patch, or a scalar (one value,
replicated). An ELL slot array (nCells, S) is S cell vectors; a vector or tensor field is one vector per component. So
a traced array `T` is a space plus a small object array over its trailing axes whose entries are register numbers.

Running the solver's numpy code on `T` operands records a linear program of three-address instructions:

    ('const', r, values)                 constant vector (mesh data), uploaded once
    ('imm', r, value)                    scalar constant
    ('ew', r, op, a[, b[, c]])           elementwise; scalar-space operands broadcast
    ('gather', r, parts)                 r[dst] = src[pos] for (src, dst, pos) in parts, 0 elsewhere (constant maps)
    ('reduce', r, kind, a)               scalar = sum / max / min over a's space
    ('copy', r, a)                       r = a (loop-carried variables)
    ('label', L) ('jump', L) ('jumpif', c, L)   control flow: jump when scalar c != 0
    ('out', name, r)                     a scalar the host reads (residuals, iteration counts)

The algorithm stays written once, in solver/ttcfd: the trace is what it does, in its own operation order.
"""
import numbers
import sys

import numpy as np


class Space:
    def __init__(self, name, n, cells=None, stack=None):
        self.name, self.n = name, n
        self.cells = cells          # patch spaces: the owning cell of every entry (faceCells)
        # stack = (base, S, w): S copies of `base` back to back, entry k * base.n + i on base entry i's core at lane
        # k * w + (i's lane in base) (Layout); bsum folds the S blocks into one base vector
        self.stack = stack

    def __repr__(self):
        return f"Space({self.name}, {self.n})"


SCALAR = Space("scalar", 1)
# run_mesh --hlate: x[idx] gathers component by component (every slot of one component, then the next), so a
# component's halo is dead after its last slot instead of live across all of them (Machine(hlate=True))
FIELD_MAJOR = False
# census only: Tracer.src[r] = the solver source line (file:line function) that created register r
SRC = False

EW_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "max": 2, "min": 2, "pow": 2,
            "gt": 2, "ge": 2, "lt": 2, "le": 2, "eq": 2, "ne": 2,
            "neg": 1, "abs": 1, "sqrt": 1, "tanh": 1, "log": 1, "exp": 1, "where": 3}


class Tracer:
    def __init__(self, dtype=np.float32):
        self.dtype = np.dtype(dtype)   # float32 for the device; float64 checks the trace itself against the solver
        self.code = []
        self.space = []             # space of every register
        self._imm = {}
        self._const = {}
        self._label = 0
        self.trips = {}             # loop label -> typical passes (placement hint only, see Machine._place)
        self.index_space = {}       # id(index array) -> Space of the gather result
        self._arange = {}
        self.slotfold = {}          # register -> k: it is step k of a fold over ELL slots (program.TracedOps._fold)
        self.src = {}               # SRC: register -> the innermost solver frame that created it

    @staticmethod
    def _where():
        f = sys._getframe(2)
        while f is not None:
            fn = f.f_code.co_filename
            if "/ttfluid/solver/" in fn or ("/ttfluid/resident/" in fn and not fn.endswith("trace.py")):
                return "%s:%d %s" % (fn.rsplit("/", 1)[1], f.f_lineno, f.f_code.co_name)
            f = f.f_back
        return "?"

    # ---- registers -----------------------------------------------------------------------------------------------
    def reg(self, space):
        self.space.append(space)
        if SRC:
            self.src[len(self.space) - 1] = self._where()
        return len(self.space) - 1

    def imm(self, v):
        v = float(self.dtype.type(v))
        key = self.dtype.type(v).tobytes()
        if key not in self._imm:
            r = self.reg(SCALAR)
            self.code.append(("imm", r, v))
            self._imm[key] = r
        return self._imm[key]

    def const(self, values, space):
        v = np.ascontiguousarray(values, self.dtype).ravel()
        assert v.size == space.n, (v.size, space)
        if v.size and np.all(v == v[0]) and not np.signbit(v[0]):
            return self.imm(v[0])
        key = (space.name, v.tobytes())
        if key not in self._const:
            r = self.reg(space)
            self.code.append(("const", r, v))
            self._const[key] = r
        return self._const[key]

    def ew(self, op, *args):
        sp = [self.space[a] for a in args]
        s = next((x for x in sp if x is not SCALAR), SCALAR)
        assert all(x is SCALAR or x is s for x in sp), (op, sp)
        r = self.reg(s)
        self.code.append(("ew", r, op) + tuple(args))
        return r

    def arange(self, n):
        """np.arange(n), one shared array per n: a gather's destination, so equal gathers carry equal index arrays
        (machine.cse matches them by identity)."""
        a = self._arange.get(n)
        if a is None:
            a = self._arange[n] = np.arange(n)
        return a

    def gather(self, space, parts):
        r = self.reg(space)
        self.code.append(("gather", r, parts))
        return r

    def bsum(self, a):
        """sum of the S blocks of a stacked-space register: a base-space register, no data leaves the core"""
        r = self.reg(self.space[a].stack[0])
        self.code.append(("bsum", r, a))
        return r

    def reduce(self, kind, a):
        r = self.reg(SCALAR)
        self.code.append(("reduce", r, kind, a))
        return r

    def copy(self, dst, src):
        self.code.append(("copy", dst, src))

    def new_label(self):
        self._label += 1
        return self._label

    def label(self, L):
        self.code.append(("label", L))

    def jump(self, L):
        self.code.append(("jump", L))

    def jumpif(self, c, L):
        self.code.append(("jumpif", c, L))

    def out(self, name, r):
        self.code.append(("out", name, r))


# =================================================================================================================
# the traced array
# =================================================================================================================

_UFUNC = {np.add: "add", np.subtract: "sub", np.multiply: "mul", np.true_divide: "div", np.divide: "div",
          np.maximum: "max", np.minimum: "min", np.power: "pow", np.greater: "gt", np.greater_equal: "ge",
          np.less: "lt", np.less_equal: "le", np.equal: "eq", np.not_equal: "ne", np.negative: "neg",
          np.absolute: "abs", np.sqrt: "sqrt", np.tanh: "tanh", np.log: "log", np.exp: "exp",
          np.logical_and: "mul", np.logical_or: "max"}


def _obj(shape, fill=None):
    a = np.empty(shape, object)
    if fill is not None:
        a.fill(fill)
    return a


class T:
    """A traced array: leading axis over `space`, trailing axes expanded into registers `c`."""
    __array_priority__ = 1000

    def __init__(self, tr, space, c):
        self.tr, self.space = tr, space
        self.c = c if isinstance(c, np.ndarray) else np.array(c, object)
        self.dtype = tr.dtype

    # ---- shape ---------------------------------------------------------------------------------------------------
    @property
    def shape(self):
        return self.c.shape if self.space is SCALAR else (self.space.n,) + self.c.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def astype(self, *a, **k):
        return self

    def copy(self):
        return T(self.tr, self.space, self.c.copy())

    def __bool__(self):
        raise TypeError("data-dependent Python branch on a traced value")

    def __float__(self):
        raise TypeError("float() of a traced value")

    def __repr__(self):
        return f"T({self.space.name}, trailing {self.c.shape})"

    # ---- lifting operands ----------------------------------------------------------------------------------------
    def _lift(self, x):
        """Any operand -> (space, object array over trailing axes)."""
        tr = self.tr
        if isinstance(x, T):
            return x.space, x.c
        if isinstance(x, (numbers.Number, np.generic)) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return SCALAR, np.array(tr.imm(float(x)), object)
        x = np.asarray(x)
        sp = self.space
        if sp is not SCALAR and x.ndim >= 1 + self.c.ndim and x.shape[0] == sp.n and x.ndim >= 1:
            # leading axis is this space (numpy aligns from the right: only when the ranks say so)
            if x.ndim == len(self.shape) or x.ndim > self.c.ndim:
                c = _obj(x.shape[1:])
                # one transposing pass, then a contiguous column per constant (a strided column copy each was
                # 29 % of a 65M chip's trace)
                xt = np.ascontiguousarray(np.moveaxis(x, 0, -1), tr.dtype) if x.ndim > 1 else x
                for idx in np.ndindex(*x.shape[1:]):
                    c[idx] = tr.const(xt[idx].copy() if x.ndim > 1 else x, sp)
                return sp, c
        # no leading axis: trailing constants (immediates)
        c = _obj(x.shape)
        for idx in np.ndindex(*x.shape):
            c[idx] = tr.imm(float(x[idx]))
        return SCALAR, c

    @staticmethod
    def _space_of(*sps):
        s = next((x for x in sps if x is not SCALAR), SCALAR)
        assert all(x is SCALAR or x is s for x in sps), sps
        return s

    def _apply(self, op, *operands):
        lifted = [self._lift(o) for o in operands]
        sp = self._space_of(*[l[0] for l in lifted])
        # leading-less constants broadcast against trailing axes only; a full-rank operand's trailing shape is c
        cs = [l[1] for l in lifted]
        shp = np.broadcast_shapes(*[c.shape for c in cs])
        cs = [np.broadcast_to(c, shp) for c in cs]
        out = _obj(shp)
        for idx in np.ndindex(*shp):
            out[idx] = self.tr.ew(op, *[c[idx] for c in cs])
        return T(self.tr, sp, out)

    # ---- arithmetic ------------------------------------------------------------------------------------------------
    def __add__(self, o): return self._apply("add", self, o)
    def __radd__(self, o): return self._apply("add", o, self)
    def __sub__(self, o): return self._apply("sub", self, o)
    def __rsub__(self, o): return self._apply("sub", o, self)
    def __mul__(self, o): return self._apply("mul", self, o)
    def __rmul__(self, o): return self._apply("mul", o, self)
    def __truediv__(self, o): return self._apply("div", self, o)
    def __rtruediv__(self, o): return self._apply("div", o, self)
    def __pow__(self, o): return self._apply("pow", self, o)
    def __neg__(self): return self._apply("neg", self)
    def __abs__(self): return self._apply("abs", self)
    def __gt__(self, o): return self._apply("gt", self, o)
    def __ge__(self, o): return self._apply("ge", self, o)
    def __lt__(self, o): return self._apply("lt", self, o)
    def __le__(self, o): return self._apply("le", self, o)
    def __and__(self, o): return self._apply("mul", self, o)
    def __rand__(self, o): return self._apply("mul", o, self)
    def __or__(self, o): return self._apply("max", self, o)
    def __invert__(self): return self._apply("sub", 1.0, self)

    def __matmul__(self, o):
        return (self * o).sum()

    # ---- indexing ----------------------------------------------------------------------------------------------------
    def _split_key(self, key):
        """-> (leading index or None for ':', trailing key)."""
        if not isinstance(key, tuple):
            key = (key,)
        if self.space is SCALAR:
            return None, key
        if key and key[0] is Ellipsis:
            n_ell = self.c.ndim - (len(key) - 1 - sum(k is None for k in key[1:]))
            if n_ell >= 0 and len(key) > 1:
                return None, (Ellipsis,) + key[1:]
            return None, key
        if not key:
            return None, ()
        k0 = key[0]
        if isinstance(k0, slice) and k0 == slice(None):
            return None, key[1:]
        return k0, key[1:]

    def __getitem__(self, key):
        lead, rest = self._split_key(key)
        if lead is None:
            return T(self.tr, self.space, self.c[rest] if rest else self.c.copy())
        if rest:
            raise NotImplementedError(f"leading index with trailing key {key}")
        idx = np.asarray(lead)
        if idx.dtype == bool:
            raise NotImplementedError("boolean mask on the leading (mesh) axis")
        return self._gather(idx)

    def _gather(self, idx):
        """x[idx] for a constant integer array idx of shape (m, ...): the result's leading axis is idx's first."""
        tr = self.tr
        sp = tr.index_space.get(id(idx))
        if sp is None:
            sp = _space_by_len(tr, idx.shape[0])
        out = _obj(idx.shape[1:] + self.c.shape)
        dst = tr.arange(idx.shape[0])
        js, cs = list(np.ndindex(*idx.shape[1:])), list(np.ndindex(*self.c.shape))
        for j, ci in ([(j, ci) for ci in cs for j in js] if FIELD_MAJOR else [(j, ci) for j in js for ci in cs]):
            out[j + ci] = tr.gather(sp, [(self.c[ci], dst, idx[(slice(None),) + j])])
        return T(tr, sp, out)

    def __setitem__(self, key, value):
        lead, rest = self._split_key(key)
        if lead is None:
            sp, vc = self._lift(value)
            assert sp is SCALAR or sp is self.space
            if sp is SCALAR and self.space is not SCALAR:   # a scalar written into a vector of this space
                vc = _map(vc, lambda r: _bcast(self.tr, r, self.space))
            # element by element: assigning an object array into an object array can nest it as one element
            pos = np.arange(self.c.size).reshape(self.c.shape)
            sel = np.atleast_1d(pos[rest] if rest else pos)
            src = np.broadcast_to(vc, np.shape(pos[rest] if rest else pos))
            src = np.atleast_1d(src) if src.ndim == 0 else src
            new = self.c.copy()
            flat = new.reshape(-1)
            for p, v in zip(sel.ravel(), src.ravel()):
                flat[p] = v
            self.c = flat.reshape(self.c.shape)
            return
        if rest:
            raise NotImplementedError(f"leading index with trailing key {key}")
        # x[cells] = values: values (same length as cells) land on those entries; the last write of a duplicate wins
        idx = np.asarray(lead)
        assert idx.ndim == 1 and idx.dtype != bool
        last = {}
        for i, c in enumerate(idx.tolist()):
            last[c] = i
        cells = np.fromiter(last.keys(), np.int64, len(last))
        pos = np.fromiter(last.values(), np.int64, len(last))
        mask = np.zeros(self.space.n, self.tr.dtype)
        mask[cells] = 1
        vs, vc = self._lift(value)
        tr = self.tr
        new = self.c.copy()
        for ci in np.ndindex(*self.c.shape):
            v = vc[ci] if vc.shape == self.c.shape else np.broadcast_to(vc, self.c.shape)[ci]
            if vs is SCALAR:
                v = _bcast(tr, v, self.space)
                g = v
            else:
                g = tr.gather(self.space, [(v, cells, pos)])
            new[ci] = tr.ew("where", tr.const(mask, self.space), g, self.c[ci])
        self.c = new

    # ---- shape manipulation --------------------------------------------------------------------------------------------
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if self.space is SCALAR:
            return T(self.tr, SCALAR, self.c.reshape(shape))
        assert shape[0] == self.space.n, shape
        return T(self.tr, self.space, self.c.reshape(shape[1:]))

    def _trail_axis(self, ax):
        nd = self.ndim
        ax = ax + nd if ax < 0 else ax
        lead = 0 if self.space is SCALAR else 1
        assert ax >= lead, "reduction over the mesh axis needs .sum() / .max() / .min() without axis"
        return ax - lead

    def sum(self, axis=None):
        if axis is None:
            return self._reduce("sum")
        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(sorted(self._trail_axis(a) for a in axes))
        c = self.c
        moved = np.moveaxis(c, axes, range(c.ndim - len(axes), c.ndim))
        lead_shape = moved.shape[:c.ndim - len(axes)]
        flat = moved.reshape(lead_shape + (-1,))
        out = _obj(lead_shape)
        for idx in np.ndindex(*lead_shape):
            regs = list(flat[idx])
            acc = regs[0]
            for r in regs[1:]:
                acc = self.tr.ew("add", acc, r)
            out[idx] = acc
        return T(self.tr, self.space, out)

    def _trail_fold(self, axis, op):
        ax = self._trail_axis(axis)
        c = np.moveaxis(self.c, ax, -1)
        out = _obj(c.shape[:-1])
        for idx in np.ndindex(*c.shape[:-1]):
            regs = list(c[idx])
            acc = regs[0]
            for r in regs[1:]:
                acc = self.tr.ew(op, acc, r)
            out[idx] = acc
        return T(self.tr, self.space, out)

    def max(self, axis=None):
        return self._reduce("max") if axis is None else self._trail_fold(axis, "max")

    def min(self, axis=None):
        return self._reduce("min") if axis is None else self._trail_fold(axis, "min")

    def mean(self, axis=None):
        if axis is None:
            return self._reduce("sum") / float(self.size)
        n = self.shape[axis]
        return self._trail_fold(axis, "add") / float(n)

    def _reduce(self, kind):
        """Whole-array reduction to a scalar: per component over the mesh axis, then over components in order."""
        tr = self.tr
        regs = [tr.reduce(kind, r) if self.space is not SCALAR else r for r in self.c.ravel()]
        acc = regs[0]
        for r in regs[1:]:
            acc = tr.ew({"sum": "add", "max": "max", "min": "min"}[kind], acc, r)
        return T(tr, SCALAR, np.array(acc, object))

    # ---- numpy protocols -----------------------------------------------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method == "at" and ufunc is np.add:
            a, idx, vals = inputs
            if not isinstance(a, T):
                raise TypeError("np.add.at into a host array with traced values: make the target traced")
            a._add_at(np.asarray(idx), vals)
            return None
        if method != "__call__" or kw.get("out") is not None:
            raise NotImplementedError(f"{ufunc}.{method} {kw}")
        op = _UFUNC.get(ufunc)
        if op is None:
            raise NotImplementedError(f"ufunc {ufunc}")
        t = next(i for i in inputs if isinstance(i, T))
        return t._apply(op, *inputs)

    def _add_at(self, idx, vals):
        """np.add.at(self, idx, vals): sequential in index order, lowered to layers of gathers so every cell's
        updates add in the order numpy applies them, with no scatter on the device."""
        tr = self.tr
        vs, vc = self._lift(vals)
        occ = {}
        layers = []
        for i, c in enumerate(idx.tolist()):
            k = occ.get(c, 0)
            occ[c] = k + 1
            if k == len(layers):
                layers.append(([], []))
            layers[k][0].append(c)
            layers[k][1].append(i)
        layers += [([], [])] * (host_max(len(layers)) - len(layers))
        new = self.c.copy()
        for ci in np.ndindex(*self.c.shape):
            v = np.broadcast_to(vc, self.c.shape)[ci]
            acc = new[ci]
            for cells, pos in layers:
                if vs is SCALAR:
                    m = np.zeros(self.space.n, tr.dtype)
                    m[cells] = 1
                    g = tr.ew("mul", tr.const(m, self.space), v)
                else:
                    g = tr.gather(self.space, [(v, np.array(cells, np.int64), np.array(pos, np.int64))])
                acc = tr.ew("add", acc, g)
            new[ci] = acc
        self.c = new

    def __array_function__(self, func, types, args, kwargs):
        f = _FUNCS.get(func)
        if f is None:
            raise NotImplementedError(f"numpy function {func.__name__} on traced arrays")
        return f(*args, **kwargs)


def host_max(n):
    """A count the trace takes from host data (np.add.at's layers): the SPMD lowering (lower/spmd.py) replaces it with
    the maximum over every chip's process."""
    return n


def _map(c, f):
    out = _obj(c.shape)
    for idx in np.ndindex(*c.shape):
        out[idx] = f(c[idx])
    return out


def _bcast(tr, r, space):
    """A scalar register as a vector over `space` (only where a whole vector must exist, e.g. stored state)."""
    return r if tr.space[r] is space else tr.ew("add", _zero_vec(tr, space), r)


def _zero_vec(tr, space, v=0.0):
    key = ("zero", space.name) if v == 0 else ("fill", space.name, v)
    if key not in tr._const:
        r = tr.reg(space)
        tr.code.append(("const", r, np.full(space.n, v, tr.dtype)))
        tr._const[key] = r
    return tr._const[key]


def _space_by_len(tr, n):
    hits = [s for s in set(tr.space) if s.n == n and s is not SCALAR]
    names = {s.name for s in hits}
    if len(names) != 1:
        raise ValueError(f"cannot tell the space of a gather of length {n}: {names}")
    return hits[0]


def _first_t(*xs):
    for x in xs:
        if isinstance(x, T):
            return x
        if isinstance(x, (list, tuple)):
            t = _first_t(*x)
            if t is not None:
                return t
    return None


def _where(c, a=None, b=None):
    t = _first_t(c, a, b)
    return t._apply("where", c, a, b)


def _broadcast_to(x, shape, subok=False):
    if x.space is SCALAR and len(shape) and not isinstance(shape, tuple):
        shape = tuple(shape)
    lead = 0 if x.space is SCALAR else 1
    if x.space is not SCALAR:
        assert shape[0] == x.space.n
    return T(x.tr, x.space, np.broadcast_to(x.c, tuple(shape[lead:])).copy())


def _swapaxes(x, a, b):
    return T(x.tr, x.space, np.swapaxes(x.c, x._trail_axis(a), x._trail_axis(b)).copy())


def _stack(xs, axis=0):
    t = _first_t(xs)
    lifted = [t._lift(x) for x in xs]
    sp = T._space_of(*[l[0] for l in lifted])
    ax = axis + (len(t.shape) + 1) if axis < 0 else axis
    assert sp is SCALAR or ax >= 1, "stacking along the mesh axis"
    c = np.stack([l[1] for l in lifted], axis=ax - (0 if sp is SCALAR else 1))
    return T(t.tr, sp, c)


def _norm(x, ord=None, axis=None):
    assert ord is None and axis is not None
    s = (x * x).sum(axis=axis)
    return np.sqrt(s)


def _full_like(x, v, dtype=None):
    z = _zero_vec(x.tr, x.space, v) if x.space is not SCALAR else x.tr.imm(v)
    return T(x.tr, x.space, _obj(x.c.shape, z))


def _zeros_like(x, dtype=None):
    return _full_like(x, 0.0)


def _ones_like(x, dtype=None):
    return _full_like(x, 1.0)


def _copy(x, *a, **k):
    return x.copy()


def _sum(x, axis=None, **k):
    return x.sum(axis=axis)


def _abs(x):
    return abs(x)


def _ndim(x):
    return x.ndim


def _shape(x):
    return x.shape


_FUNCS = {np.where: _where, np.broadcast_to: _broadcast_to, np.swapaxes: _swapaxes, np.stack: _stack,
          np.linalg.norm: _norm, np.zeros_like: _zeros_like, np.ones_like: _ones_like, np.copy: _copy, np.sum: _sum, np.ndim: _ndim,
          np.shape: _shape}
