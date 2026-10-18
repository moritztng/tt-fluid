"""Volume fields with OpenFOAM boundary conditions.

A field is its cell values plus one boundary condition per non-empty patch. Each condition exposes the four
coefficient sets fvm needs (value/gradient x internal/boundary coefficients, as in fvPatchField) and evaluates its
patch values from the cell values.
"""
import re

import numpy as np


class BC:
    """Base: a patch field. `value` holds the current patch values, shape (nFaces,) or (nFaces, 3)."""
    fixes_value = False   # fixedValue-like (pressure needs no reference)
    assignable = True     # fvPatchField::assignable(): False -> constrainHbyA sets HbyA's patch value to U's
    coupled = False

    def __init__(self, field, patch, entry):
        self.field, self.patch, self.entry = field, patch, entry
        w = field.width
        shape = (patch.size,) if w == 1 else (patch.size, w)
        v = entry.get("value")
        self.value = np.broadcast_to(np.asarray(v, float), shape).copy() if v is not None else np.zeros(shape)

    def internal(self):
        return self.field.internal[self.patch.faceCells]

    def evaluate(self):
        pass

    def snGrad(self):
        return (self.value - self.internal()) * self._dc()

    def _dc(self):
        dc = self.patch.deltaCoeffs
        return dc if self.field.width == 1 else dc[:, None]

    def _ones(self, x):
        return np.full((self.patch.size,) if self.field.width == 1 else (self.patch.size, self.field.width), x, float)


class FixedValue(BC):
    fixes_value = True
    assignable = False

    def valueInternalCoeffs(self, w):
        return self._ones(0.0)

    def valueBoundaryCoeffs(self, w):
        return self.value.copy()

    def gradientInternalCoeffs(self):
        return -self._ones(1.0) * self._dc()

    def gradientBoundaryCoeffs(self):
        return self._dc() * self.value


class NoSlip(FixedValue):
    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        self.value[:] = 0.0


class UniformFixedValue(FixedValue):
    """uniformFixedValue with a constant Function1 (`uniformValue constant X` or `uniformValue X`): fixedValue X."""

    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        u = entry["uniformValue"]
        t = [x for x in (u if isinstance(u, list) else [u]) if x not in ("(", ")")]
        if isinstance(t[0], str) and not _isnum(t[0]):
            if t[0] != "constant":
                raise NotImplementedError(f"uniformFixedValue {t[0]} on {patch.name}")
            t = t[1:]
        self.value[:] = np.asarray([float(str(x).strip("()")) for x in t], float).reshape(self.value.shape[1:])


def _isnum(x):
    try:
        float(str(x).strip("()"))
        return True
    except ValueError:
        return False


class FixedGradient(BC):
    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        g = entry.get("gradient", 0.0)
        self.gradient = np.broadcast_to(np.asarray(g, float), self.value.shape).copy()
        self.evaluate()

    def evaluate(self):
        self.value = self.internal() + self.gradient / self._dc()

    def snGrad(self):
        return self.gradient.copy()

    def valueInternalCoeffs(self, w):
        return self._ones(1.0)

    def valueBoundaryCoeffs(self, w):
        return self.gradient / self._dc()

    def gradientInternalCoeffs(self):
        return self._ones(0.0)

    def gradientBoundaryCoeffs(self):
        return self.gradient.copy()


class ZeroGradient(FixedGradient):
    pass


class Calculated(BC):
    """Values set by whoever owns the field (nut); never used to build a matrix."""
    def valueInternalCoeffs(self, w):
        raise TypeError(f"calculated patch {self.patch.name} of {self.field.name} cannot build a matrix")

    valueBoundaryCoeffs = gradientInternalCoeffs = gradientBoundaryCoeffs = valueInternalCoeffs


class SlipVector(BC):
    """slip on a vector (basicSymmetry through transformFvPatchField): the patch keeps the tangential part of the
    cell value, snGrad = (transform(I - 2 n n, U_i) - U_i) deltaCoeffs / 2 = -n (n . U_i) deltaCoeffs, and the matrix
    takes the normal part implicitly through snGradTransformDiag = |n| per component. Not assignable
    (slipFvPatchField::assignable() is false), so constrainHbyA gives HbyA the slip value: no flux through it."""

    assignable = False

    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        self.n = patch.nf
        self.diag = np.abs(patch.nf)
        self.evaluate()

    def _normal(self):
        iF = self.internal()
        return self.n * (iF * self.n).sum(axis=1)[:, None]

    def evaluate(self):
        self.value = self.internal() - self._normal()

    def snGrad(self):
        return -self._normal() * self._dc()

    def valueInternalCoeffs(self, w):
        return 1.0 - self.diag

    def valueBoundaryCoeffs(self, w):
        return self.value - (1.0 - self.diag) * self.internal()

    def gradientInternalCoeffs(self):
        return -self._dc() * self.diag

    def gradientBoundaryCoeffs(self):
        return self.snGrad() + self._dc() * self.diag * self.internal()


def slip(field, patch, entry):
    """slip: zeroGradient for a scalar (basicSymmetryFvPatchScalarField), the transform condition for a vector."""
    return ZeroGradient(field, patch, entry) if field.width == 1 else SlipVector(field, patch, entry)


class InletOutlet(ZeroGradient):
    """inletOutlet where every face flows out: zeroGradient, which is what OpenFOAM's mixed condition reduces to at
    valueFraction = 1 - pos0(phi) = 0. The solver checks the outflow on its host path (SimpleFoam.check_outflow);
    a face with inflow would need the fixed inletValue, which this class does not model."""

    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        self.inletValue = entry.get("inletValue")


REGISTRY = {
    "slip": slip,
    "symmetry": slip,          # symmetryFvPatchField and slipFvPatchField are both basicSymmetryFvPatchField
    "inletOutlet": InletOutlet,
    "fixedValue": FixedValue,
    "uniformFixedValue": UniformFixedValue,
    "noSlip": NoSlip,
    "zeroGradient": ZeroGradient,
    "fixedGradient": FixedGradient,
    "calculated": Calculated,
    "extrapolatedCalculated": Calculated,
    # wall-function conditions whose matrix role is a plain one; their physics lives in the turbulence model
    "kqRWallFunction": ZeroGradient,
}


# patch types whose polyPatch adds its own type to inGroups (polyPatch.C constraintType -> addGroup; wallPolyPatch.C
# adds "wall")
_SELF_GROUP = {"wall", "empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicAMI", "cyclicACMI",
               "cyclicSlip", "processor", "processorCyclic", "nonuniformTransformCyclic", "overset"}
_REGEX_CHARS = re.compile(r"[.*+?()|\[\]{}^$\\]")


def _groups(p):
    g = p.dict.get("inGroups")
    if isinstance(g, str):
        g = [g]
    names = [t for t in (g or []) if t not in "()" and not t.isdigit()]
    names = [n.split("(", 1)[-1].rstrip(")") for n in names if n.split("(", 1)[-1].rstrip(")")]
    if p.type in _SELF_GROUP and p.type not in names:
        names.append(p.type)
    return names


def _entry_key(entries, p):
    """Which boundaryField entry a patch takes, GeometricBoundaryField::readField (src/OpenFOAM/fields/
    GeometricFields/GeometricField/GeometricBoundaryField.C): 1. a literal key equal to the patch name; 2. a literal
    key naming one of the patch's groups, the last such key in the dictionary winning; 3. dict.findDict(name):
    an exact key, then the regex keys ("side.*") last-declared first (dictionarySearch.C csearch). Quoted keys are
    regex (keyType), so they never match in steps 1-2."""
    patterns = getattr(entries, "patterns", None)
    if patterns is None:      # a plain dict: treat keys with regex characters as the quoted ones
        patterns = [k for k in entries if _REGEX_CHARS.search(k)]
    literal = [k for k in entries if k not in patterns]
    if p.name in literal:
        return p.name
    groups = _groups(p)
    for k in reversed(literal):
        if k in groups:
            return k
    if p.name in entries:
        return p.name
    for k in reversed(patterns):
        if re.fullmatch(k, p.name):
            return k
    raise KeyError(f"no boundaryField entry for patch {p.name} (groups {groups})")


def _entry(entries, p, width=1):
    if p.parts:      # a merged run: the members' common entry, their values concatenated face by face
        es = [_entry(entries, q, width) for q in p.parts]
        e = dict(es[0])
        if "value" in e:
            e["value"] = np.concatenate([np.broadcast_to(np.asarray(x["value"], float),
                                                         (q.size,) if width == 1 else (q.size, width))
                                         for x, q in zip(es, p.parts)])
        return e
    return entries[_entry_key(entries, p)]


def _canon(e):
    return tuple(sorted((k, repr(v)) for k, v in e.items() if k != "value"))


def group_runs(mesh, entries):
    """Runs of consecutive non-empty patches of one type whose entries agree in every field except their values:
    [(names, new name)], for Mesh.merge_patches. `entries`: one boundaryField dict per field."""
    runs, cur = [], None
    for p in mesh.patches:
        sig = None if p.type in ("empty", "cyclic", "processor") else \
            (p.type,) + tuple(_canon(_entry(e, p)) for e in entries)
        if sig is not None and cur is not None and cur[0] == sig:
            cur[1].append(p.name)
        else:
            cur = (sig, [p.name])
            runs.append(cur)
    return [(names, f"{names[0]}+{len(names) - 1}") for sig, names in runs if sig is not None and len(names) > 1]


class VolField:
    def __init__(self, name, mesh, internal, entries, dims=None, cls=None, registry=None):
        self.name, self.mesh = name, mesh
        self.internal = np.asarray(internal, float)
        self.width = 1 if self.internal.ndim == 1 else self.internal.shape[1]
        self.dims, self.cls = dims, cls or ("volScalarField" if self.width == 1 else "volVectorField")
        reg = dict(REGISTRY, **(registry or {}))
        self.bc = {}
        for p in mesh.patches:
            if p.type == "empty":
                continue
            e = _entry(entries, p, self.width)
            self.bc[p.name] = reg[e["type"]](self, p, e)
        self.entries = entries

    @classmethod
    def read(cls, mesh, path, registry=None):
        from . import foamio
        d = foamio.read_field(path)
        n = mesh.nCells
        internal = d["internal"]
        if np.ndim(internal) == 0 or (d["width"] > 1 and np.ndim(internal) == 1):
            internal = np.broadcast_to(np.asarray(internal, float), (n,) if d["width"] == 1 else (n, d["width"])).copy()
        entries = foamio.FoamDict()
        entries.patterns = list(getattr(d["boundary"], "patterns", []))
        for k, e in d["boundary"].items():
            e = dict(e)
            for key, v in e.items():   # $internalField macro
                if isinstance(v, str) and v == "$internalField":
                    e[key] = d["internal"]
            entries[k] = e
        return cls(path.name if hasattr(path, "name") else str(path), mesh, internal, entries,
                   dims=d["dimensions"], cls=d["cls"], registry=registry)

    def correct_bc(self):
        for b in self.bc.values():
            b.evaluate()

    def boundary_items(self):
        return self.bc.items()

    def write(self, case, time, name=None):
        from . import foamio
        boundary = {}
        for p in self.mesh.patches:
            if p.type == "empty":
                boundary[p.name] = {"type": "empty"}
                continue
            b = self.bc[p.name]
            e = {"type": _entry(self.entries, p, self.width)["type"]}
            for k, v in _entry(self.entries, p, self.width).items():
                uniform = isinstance(v, np.ndarray) and (v.ndim == 0 or (self.width > 1 and v.shape == (self.width,)))
                if k not in ("type", "value") and not isinstance(v, dict) and (uniform or not isinstance(v, np.ndarray)):
                    e[k] = v    # inletValue uniform (0 0 0) and the like; per-face arrays are rewritten below
            if hasattr(b, "gradient") and not isinstance(b, ZeroGradient):
                e["gradient"] = b.gradient
            if not isinstance(b, ZeroGradient) or isinstance(b, InletOutlet):     # OpenFOAM requires inletOutlet's value
                e["value"] = b.value
            off = 0
            for n, k in self.mesh.merged.get(p.name, [(p.name, p.size)]):     # a merged run is written per patch
                boundary[n] = {a: (v[off:off + k] if isinstance(v, np.ndarray) and len(v) == p.size else v)
                               for a, v in e.items()}
                off += k
        n = name or self.name
        foamio.write_field(f"{case}/{time}/{n}", n, self.cls, self.dims or [0] * 7, self.internal, boundary,
                           location=str(time))
