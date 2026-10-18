"""Read and write the OpenFOAM case files the solver needs: polyMesh and volume fields.

Covers ASCII and binary (label 32, scalar 64) files, plain and compact face lists, and the
field entries the incompressible solvers use (uniform/nonuniform values, patch dictionaries).
"""
import math
import os
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r'"[^"]*"|\$\{[^}]*\}[^\s{}();\[\]"]*|[{}();\[\]]|[^\s{}();\[\]"]+')


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _split_header(raw):
    """Return (header dict, body bytes) for a FoamFile."""
    m = re.search(rb"FoamFile\s*\{(.*?)\}", raw, re.S)
    head = {}
    if m:
        for k, v in re.findall(rb"(\w+)\s+([^;]*);", m.group(1)):
            head[k.decode()] = v.decode().strip().strip('"')
        raw = raw[m.end():]
    return head, raw


def _read_list_binary(raw, pos, n, dtype, width):
    """Binary list body starts with '(' at pos; returns array and position after ')'."""
    start = raw.index(b"(", pos) + 1
    nbytes = n * width * np.dtype(dtype).itemsize
    arr = np.frombuffer(raw[start:start + nbytes], dtype=dtype).copy()
    end = raw.index(b")", start + nbytes) + 1
    return arr.reshape(n, width) if width > 1 else arr, end


def _read_list_ascii(raw, pos, n, width, dtype):
    start = raw.index(b"(", pos) + 1
    if width == 1:
        end = start
        # find the matching ')' of a flat list
        end = raw.index(b")", start)
        arr = np.array(raw[start:end].split(), dtype=dtype)
        return arr, end + 1
    # list of tuples "(a b c)"
    depth, i = 1, start
    while depth:
        c = raw[i:i + 1]
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
        i += 1
    body = raw[start:i - 1].replace(b"(", b" ").replace(b")", b" ")
    arr = np.array(body.split(), dtype=dtype).reshape(n, width)
    return arr, i


_COUNT = re.compile(rb"(?:\s|//[^\n]*\n|/\*.*?\*/)*(\d+)\s*", re.S)


def _list_after(raw, pos, binary, dtype, width):
    """Parse 'N ( ... )' or 'N{value}' starting at pos."""
    m = _COUNT.match(raw, pos)
    n = int(m.group(1))
    p = m.end()
    if raw[p:p + 1] == b"{":
        end = raw.index(b"}", p)
        vals = raw[p + 1:end].replace(b"(", b" ").replace(b")", b" ").split()
        v = np.array(vals, dtype=dtype)
        return np.tile(v, (n, 1)) if width > 1 else np.full(n, v[0]), end + 1
    if binary:
        return _read_list_binary(raw, p, n, dtype, width)
    return _read_list_ascii(raw, p, n, width, dtype)


def read_points(path):
    raw = Path(path).read_bytes()
    head, body = _split_header(raw)
    return _list_after(body, 0, head.get("format") == "binary", np.float64, 3)[0]


def read_labels(path):
    raw = Path(path).read_bytes()
    head, body = _split_header(raw)
    return _list_after(body, 0, head.get("format") == "binary", np.int32, 1)[0].astype(np.int64)


def read_faces(path):
    """Return (offsets, flat point labels) for faceList or faceCompactList."""
    raw = Path(path).read_bytes()
    head, body = _split_header(raw)
    binary = head.get("format") == "binary"
    if head.get("class") == "faceCompactList":
        offs, p = _list_after(body, 0, binary, np.int32, 1)
        flat, _ = _list_after(body, p, binary, np.int32, 1)
        return offs.astype(np.int64), flat.astype(np.int64)
    # ascii faceList: N ( k(a b c ...) ... )
    m = re.compile(rb"(?:\s|//[^\n]*\n|/\*.*?\*/)*(\d+)\s*\(", re.S).match(body)
    n = int(m.group(1))
    toks = body[m.end():].replace(b"(", b" ").replace(b")", b" ").split()
    vals = np.array(toks[: _faces_token_count(toks, n)], dtype=np.int64)
    sizes = np.empty(n, np.int64)
    flat = np.empty(len(vals) - n, np.int64)
    i = j = 0
    for f in range(n):
        k = vals[i]
        sizes[f] = k
        flat[j:j + k] = vals[i + 1:i + 1 + k]
        i += k + 1
        j += k
    offs = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=offs[1:])
    return offs, flat


def _faces_token_count(toks, n):
    i = 0
    for _ in range(n):
        i += int(toks[i]) + 1
    return i


class FoamDict(dict):
    """An OpenFOAM dictionary: keys (quotes stripped) map to token lists or sub-dictionaries, in file order.
    `patterns` lists the keys that were quoted, which OpenFOAM reads as regular expressions (keyType(string) is
    REGEX, src/OpenFOAM/primitives/strings/keyType/keyTypeI.H); `parent` is the enclosing dictionary."""

    def __init__(self, *a, parent=None, **kw):
        super().__init__(*a, **kw)
        self.parent = parent
        self.patterns = []

    def top(self):
        d = self
        while d.parent is not None:
            d = d.parent
        return d

    def find(self, key, recursive=False):
        """dictionary::csearch (src/OpenFOAM/db/dictionary/dictionarySearch.C): the exact key, then the patterns
        last-declared first (patterns_ is push_front, dictionary.C), then (recursive) the parent's."""
        d = self
        while d is not None:
            if key in d:
                return d[key]
            for k in reversed(d.patterns):
                if re.fullmatch(k, key):
                    return d[k]
            d = d.parent if recursive else None
        return None

    def find_scoped(self, var):
        """dictionary::csearchScoped (dictionarySearch.C): ':a.b' from the top, '..a' one level up per extra dot,
        'a.b' with the first word searched through the parents; '/'-scoped names likewise."""
        if var.startswith(":"):
            d, parts = self.top(), var[1:].split(".")
        elif var.startswith("/"):
            d, parts = self.top(), [p for p in var.split("/") if p]
        elif "/" in var:
            d, parts = self, var.split("/")
        elif var.startswith("."):
            d, n = self, 1
            while n < len(var) and var[n] == ".":
                d, n = d.parent, n + 1
            parts = var[n:].split(".")
        else:
            parts = var.split(".")
            v = self.find(parts[0], recursive=True)
            for p in parts[1:]:
                if not isinstance(v, FoamDict):
                    return self.find(var, recursive=True)     # a literal key containing a dot
                v = v.find(p)
            return v
        v = d
        for p in parts:
            if not isinstance(v, FoamDict):
                return None
            v = v.find(p)
        return v

    def set_entry(self, key, val, pattern=False):
        """Add with inputMode merge, the default (entryIO.C): a dictionary merges into an existing one, anything
        else replaces it."""
        old = self.get(key)
        if isinstance(old, FoamDict) and isinstance(val, FoamDict):
            for k, v in val.items():
                old.set_entry(k, v, k in val.patterns)
            return
        if isinstance(val, FoamDict):
            val.parent = self
        self[key] = val
        if pattern and key not in self.patterns:
            self.patterns.append(key)
        elif not pattern and key in self.patterns:
            self.patterns.remove(key)


_ETC_BUNDLED = Path(__file__).parent / "etc"
_VAR = re.compile(r"\$(\{[^}]*\}|:?[\w.:]*\w)")


def _etc_file(name):
    """#includeEtc: $WM_PROJECT_DIR/etc first, then the copy bundled with the solver (ttfluid/solver/etc)."""
    roots = [Path(os.environ["WM_PROJECT_DIR"]) / "etc"] if os.environ.get("WM_PROJECT_DIR") else []
    for root in roots + [_ETC_BUNDLED]:
        if (root / name).is_file():
            return root / name
    raise FileNotFoundError(f"#includeEtc {name}: not under $WM_PROJECT_DIR/etc or {_ETC_BUNDLED}")


def _lookup_tokens(scope, var, ctx):
    if var.startswith("{"):
        var = var[1:-1]
    if var == "FOAM_CASE" and ctx.get("case"):
        return [str(ctx["case"])]
    v = scope.find_scoped(var) if scope is not None else None
    if v is None:
        if var in os.environ:
            return [os.environ[var]]
        raise KeyError(f"undefined variable ${var}")
    return v


def _expand_string(s, scope, ctx):
    """stringOps::inplaceExpand: $var / ${var} inside a string become the entry's tokens, space-joined."""
    def rep(m):
        v = _lookup_tokens(scope, m.group(1), ctx)
        if isinstance(v, FoamDict):
            raise ValueError(f"${m.group(1)} is a dictionary, cannot expand it inside a string")
        return " ".join(v)
    return _VAR.sub(rep, s)


_EVAL_NS = {"sqr": lambda x: x * x, "magSqr": lambda x: x * x, "sqrt": math.sqrt, "pow": pow, "mag": abs,
            "exp": math.exp, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "max": max, "min": min, "pi": lambda: math.pi}


def _eval(expr):
    """#eval of a scalar expression (functionEntries/evalEntry): the arithmetic the cases use."""
    if not re.fullmatch(r"[\w\s.+\-*/(),]*", expr):
        raise NotImplementedError(f"#eval {expr!r}")
    return repr(float(eval(expr, {"__builtins__": {}}, _EVAL_NS)))   # noqa: S307 - charset-checked arithmetic


def _include_tokens(path):
    toks = _tokens(_strip_comments(Path(path).read_text(errors="replace")))
    if toks[:2] == ["FoamFile", "{"]:
        toks = toks[toks.index("}") + 1:]
    return [("#file", Path(path))] + toks + [("#endfile", None)]


def _expand_value(tokens, scope, ctx):
    """$var and #eval inside a primitive entry, expanded at read time as primitiveEntry does."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t == "#eval":
            if tokens[i + 1] == "{":
                j, depth = i + 1, 0
                while True:
                    depth += (tokens[j] == "{") - (tokens[j] == "}")
                    if depth == 0:
                        break
                    j += 1
                expr, i = " ".join(tokens[i + 2:j]), j + 1
            else:
                expr, i = tokens[i + 1].strip('"'), i + 2
            out.append(_eval(_expand_string(expr, scope, ctx)))
            continue
        m = _VAR.fullmatch(t) if t.startswith("$") else None
        if m:
            v = _lookup_tokens(scope, m.group(1), ctx)
            if isinstance(v, FoamDict):
                raise ValueError(f"{t}: a dictionary used as a value")
            out.extend(v)
        else:
            out.append(_expand_string(t, scope, ctx) if "$" in t else t)
        i += 1
    return out


_INCLUDES = ("#include", "#includeIfPresent", "#sinclude", "#includeEtc", "#includeFunc", "#includeFunctions")


def _parse_dict(tokens, i, scope=None, ctx=None):
    """Parse tokens from after '{' to the matching '}' into a FoamDict, applying #include/#includeEtc, $var and
    ${..} expansion, $var keywords ($inletPatch { .. }, $p;) and #eval while reading, as OpenFOAM v2412 does."""
    ctx = ctx if ctx is not None else {"files": []}
    out = FoamDict(parent=scope)
    while True:
        t = tokens[i]
        if isinstance(t, tuple):            # include markers: the current file, for relative #include
            if t[0] == "#file":
                ctx["files"].append(t[1])
            else:
                ctx["files"].pop()
            i += 1
            continue
        if t == "}":
            return out, i
        if t in _INCLUDES:
            arg = tokens[i + 1]
            i += 2
            if t in ("#includeFunc", "#includeFunctions"):    # function objects: not part of the solve
                continue
            name = _expand_string(arg.strip('"'), out, ctx)
            if t == "#includeEtc":
                p = _etc_file(name)
            else:   # includeEntry::resolveFile: relative to the including file's directory
                cur = ctx["files"][-1] if ctx["files"] else None
                p = Path(name) if Path(name).is_absolute() or cur is None else Path(cur).parent / name
                if not p.is_file():
                    if t != "#include":
                        continue
                    raise FileNotFoundError(f"#include {name} (resolved {p})")
            tokens[i:i] = _include_tokens(p)
            continue
        if t in ("#inputMode", "#remove"):
            i += 2
            continue
        quoted = t.startswith('"')
        key = t.strip('"')
        i += 1
        if t.startswith("$") and len(t) > 1:
            v = _lookup_tokens(out, t[1:], ctx)
            if tokens[i] == "{":            # $var { .. }: the variable's value is the keyword (entryIO.C)
                if isinstance(v, FoamDict) or not v:
                    raise ValueError(f"{t} as a keyword is not a word or string")
                key, quoted = v[0].strip('"'), v[0].startswith('"')
            else:                           # $var; merges the named dictionary in (substituteScopedKeyword)
                if not isinstance(v, FoamDict):
                    raise ValueError(f"{t}; does not name a dictionary")
                for k, e in v.items():
                    out.set_entry(k, _copy(e), k in v.patterns)
                while tokens[i] != ";":
                    i += 1
                i += 1
                continue
        if tokens[i] == "{":
            sub, i = _parse_dict(tokens, i + 1, out, ctx)
            out.set_entry(key, sub, quoted)
            i += 1
            continue
        val, depth = [], 0
        while not (tokens[i] == ";" and depth == 0):
            t = tokens[i]
            if not isinstance(t, tuple):
                depth += (t == "(") - (t == ")")
                val.append(t)
            i += 1
        out.set_entry(key, _expand_value(val, out, ctx), quoted)
        i += 1


def _copy(e):
    if isinstance(e, FoamDict):
        d = FoamDict(parent=e.parent)
        for k, v in e.items():
            d[k] = _copy(v)
        d.patterns = list(e.patterns)
        return d
    return list(e)


def _tokens(text):
    """Tokens, keeping words glued to a following parenthesis group intact: div(phi,U), grad(U)."""
    out, pos = [], 0
    while True:
        m = _TOKEN.search(text, pos)
        if not m:
            return out
        v = m.start()
        if text.startswith("#{", v):      # verbatim #{ ... #} (coded function objects): one token, as OpenFOAM's verbatimString
            end = text.index("#}", v + 2) + 2
            out.append(text[v:end])
            pos = end
            continue
        t, pos = m.group(0), m.end()
        if t not in "{}();[]" and text[pos:pos + 1] == "(" and not t.startswith('"'):
            depth = 0
            while True:
                c = text[pos]
                depth += (c == "(") - (c == ")")
                pos += 1
                if depth == 0:
                    break
            t = t + text[m.end():pos]
        out.append(t)


def parse_dict_text(text, path=None, case=None):
    """Parse dictionary text. `path` (the file it came from) resolves relative #include; `case` is $FOAM_CASE."""
    toks = _tokens(_strip_comments(text))
    out, _ = _parse_dict(toks + ["}"], 0, None, {"case": case, "files": [Path(path)] if path else []})
    return out


def _case_of(path):
    """$FOAM_CASE of a file in 0/, system/ or constant/: the directory above."""
    return Path(path).resolve().parent.parent


def read_dict(path, case=None):
    """Read an OpenFOAM dictionary file (FoamFile header dropped), expanding includes and macros."""
    d = parse_dict_text(Path(path).read_text(errors="replace"), path=path, case=case or _case_of(path))
    d.pop("FoamFile", None)
    return d


def read_boundary(path):
    """Return list of (name, dict) in file order; dict has type, nFaces, startFace (ints)."""
    raw = Path(path).read_bytes()
    _, body = _split_header(raw)
    toks = _TOKEN.findall(_strip_comments(body.decode()))
    i = toks.index("(") + 1
    patches = []
    while toks[i] != ")":
        name = toks[i]
        d, i = _parse_dict(toks, i + 2)
        i += 1
        d = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in d.items()}
        d["nFaces"], d["startFace"] = int(d["nFaces"]), int(d["startFace"])
        patches.append((name, d))
    return patches


_FIELD_WIDTH = {"volScalarField": 1, "volVectorField": 3, "volSymmTensorField": 6, "volTensorField": 9,
                "surfaceScalarField": 1}


def _value(tokens, width, binary_blob=None):
    """Parse 'uniform x' / 'uniform (x y z)' / 'nonuniform List<..> N (...)' token lists."""
    if binary_blob is not None and tokens[0] == "nonuniform" and tokens[1].startswith("__bin"):
        return ("nonuniform", binary_blob[int(tokens[1][5:-2])])
    if tokens[0] == "uniform":
        nums = [float(t) for t in tokens[1:] if t not in "()"]
        return ("uniform", np.array(nums) if width > 1 else nums[0])
    if tokens[0] == "nonuniform":
        if "(" in tokens[2]:   # short lists are written on one line, "6(a b c ...)", glued by _tokens
            head, rest = tokens[2].split("(", 1)
            tokens = tokens[:2] + [head] + rest.replace("(", " ").replace(")", " ").split() + tokens[3:]
        n = int(tokens[2])
        nums = np.array([float(t) for t in tokens[3:] if t not in "()"], dtype=np.float64)
        return ("nonuniform", nums.reshape(n, width) if width > 1 else nums)
    raise ValueError(f"cannot parse field value {tokens[:4]}")


_LIST_WIDTH = {b"scalar": 1, b"vector": 3, b"symmTensor": 6, b"tensor": 9}
_BIN_LIST = re.compile(rb"nonuniform\s+List<(\w+)>\s+(\d+)\s*\(")


def _cut_binary_lists(body):
    """Binary field: every 'nonuniform List<T> N (<raw float64>)' becomes the token 'nonuniform __binK__' and the
    array goes to blobs[K], so the rest of the file parses as text."""
    out, blobs, pos = [], [], 0
    while True:
        m = _BIN_LIST.search(body, pos)
        if not m:
            break
        w, n = _LIST_WIDTH[m.group(1)], int(m.group(2))
        arr, end = _read_list_binary(body, m.end() - 1, n, np.float64, w)
        out += [body[pos:m.start()], b"nonuniform __bin%d__" % len(blobs)]
        blobs.append(arr)
        pos = end
    out.append(body[pos:])
    return b"".join(out), blobs


def read_field(path):
    """Read an ASCII or binary vol*Field. Returns dict(cls, dimensions, internal (array or scalar), boundary {patch: dict})."""
    raw = Path(path).read_bytes()
    head, body = _split_header(raw)
    blobs = None
    if head.get("format") == "binary":
        body, blobs = _cut_binary_lists(body)
    width = _FIELD_WIDTH[head["class"]]
    d = parse_dict_text(body.decode(errors="replace"), path=path, case=_case_of(path))
    dims = [float(t) for t in d["dimensions"] if t not in "[]"]
    kind, internal = _value(d["internalField"], width, blobs)
    boundary = FoamDict()
    boundary.patterns = list(d["boundaryField"].patterns)
    for name, pd in d["boundaryField"].items():
        e = {}
        for k, v in pd.items():
            if isinstance(v, dict):
                e[k] = v
            elif v and v[0] in ("uniform", "nonuniform"):
                e[k] = _value(v, width, blobs)[1]
            else:
                e[k] = v[0] if len(v) == 1 else v
        boundary[name] = e
    return {"cls": head["class"], "dimensions": dims, "internal": internal, "boundary": boundary,
            "width": width}


def _fmt_value(v, width):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 0 or (width > 1 and v.ndim == 1):
        if width == 1:
            return f"uniform {float(v):.10g}"
        return "uniform (" + " ".join(f"{x:.10g}" for x in v) + ")"
    tname = {1: "scalar", 3: "vector"}[width]
    lines = [f"nonuniform List<{tname}> ", str(len(v)), "("]
    if width == 1:
        lines += [f"{x:.10g}" for x in v]
    else:
        lines += ["(" + " ".join(f"{x:.10g}" for x in r) + ")" for r in v]
    lines.append(")")
    return "\n".join(lines)


def write_field(path, name, cls, dims, internal, boundary, location="0"):
    """Write an ASCII vol*Field that OpenFOAM tools (foamToVTK, postProcess, paraFoam) read."""
    width = _FIELD_WIDTH[cls]
    out = [
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    location    \"{location}\";\n    object      {name};\n}}\n",
        "dimensions      [" + " ".join(f"{d:g}" for d in dims) + "];\n",
        "internalField   " + _fmt_value(internal, width) + ";\n",
        "boundaryField\n{",
    ]
    for pname, e in boundary.items():
        out.append(f"    {pname}\n    {{")
        for k, v in e.items():
            if isinstance(v, str):
                out.append(f"        {k} {v};")
            elif isinstance(v, (list, tuple)):
                out.append(f"        {k} {' '.join(map(str, v))};")
            elif isinstance(v, dict):
                continue
            else:
                out.append(f"        {k} {_fmt_value(v, width)};")
        out.append("    }")
    out.append("}\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out))
