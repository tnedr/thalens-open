"""
Latent Compiler — native lowering: emit C, compile to machine code, load.

``lowering.py`` emits a NumPy executor; this module goes the last mile of the
"compile to machine code" vision. For an accepted latent it:

  1. emits a self-contained C function implementing the latent apply,
  2. compiles it to a shared object with ``clang -O3 -march=native``, and
  3. loads it back as a callable through ``ctypes``.

What native lowering buys you (measured, honestly):

  * **The algorithmic win is real and large vs the dense baseline:** rank-r
    matrix apply ~6x vs dense gemv (rank 32 of 1024); sparse Boolean apply
    ~400x vs the dense Walsh-Koopman operator. This is the latent doing its job.
  * **vs the NumPy latent executor it roughly ties at these sizes.** NumPy's
    primitives (BLAS gemv, the C-level ``np.add.at`` scatter) are already
    compiled; the ctypes per-call boundary costs ~a microsecond, so a small
    standalone C kernel does not beat them for moderate n. The win would widen
    for larger rank, batched/SIMD lowering, or fixed-size unrolled kernels.
  * **The point of native lowering is deployability, not a NumPy speed race:**
    it emits a self-contained machine-code kernel for the *fixed* discovered
    structure that runs with no Python/NumPy runtime — the artifact a hardware
    or embedded backend would consume.

It is explicitly not a claim to beat tuned BLAS on large dense GEMM.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .core import CompileResult


def compiler_available() -> str | None:
    """Return the first available C compiler executable name, or None."""
    for cc in ("clang", "gcc", "cc"):
        path = shutil.which(cc)
        # Skip the nexus `cc` shell alias which is not a C compiler.
        if path and "command_center" not in path:
            return cc
    return None


_SHARED_EXT = ".dylib" if sys.platform == "darwin" else ".so"


def emit_c_source(result: CompileResult, function_name: str = "latent_kernel") -> str:
    """Emit self-contained C source implementing the latent apply."""
    if result.basis == "walsh_koopman":
        rep = result.representation
        n = rep["shape"][0]
        return (
            "#include <string.h>\n\n"
            f"/* Sparse Walsh-Koopman latent kernel: {rep['nnz']} nonzeros, dim {n}.\n"
            "   out = K_sparse @ g, i.e. out[rows[k]] += vals[k] * g[cols[k]]. */\n"
            f"void {function_name}(const long* rows, const long* cols, const double* vals,\n"
            "                     const double* g, double* out, long nnz, long n) {\n"
            "    memset(out, 0, (size_t)n * sizeof(double));\n"
            "    for (long k = 0; k < nnz; ++k) out[rows[k]] += vals[k] * g[cols[k]];\n"
            "}\n"
        )

    if result.basis == "spectral_svd":
        rep = result.representation
        r = rep["rank"]
        n = rep["shape"][0]
        return (
            "#include <stddef.h>\n\n"
            "/* Low-rank spectral latent kernel: rank {r}, dim {n}.\n"
            "   y = U (s * (Vt x)); U is n x r row-major, Vt is r x n row-major. */\n".format(r=r, n=n)
            + f"void {function_name}(const double* U, const double* s, const double* Vt,\n"
            "                     const double* x, double* y, double* tmp, long n, long r) {\n"
            "    for (long k = 0; k < r; ++k) {\n"
            "        double acc = 0.0;\n"
            "        const double* vt = Vt + (size_t)k * n;\n"
            "        for (long j = 0; j < n; ++j) acc += vt[j] * x[j];\n"
            "        tmp[k] = acc * s[k];\n"
            "    }\n"
            "    for (long i = 0; i < n; ++i) {\n"
            "        double acc = 0.0;\n"
            "        const double* u = U + (size_t)i * r;\n"
            "        for (long k = 0; k < r; ++k) acc += u[k] * tmp[k];\n"
            "        y[i] = acc;\n"
            "    }\n"
            "}\n"
        )

    raise ValueError(f"unknown basis {result.basis!r}")


_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_C_LONG_P = ctypes.POINTER(ctypes.c_long)


def _as_c(arr: np.ndarray, dtype) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=dtype)


@dataclass
class NativeKernel:
    """A compiled latent kernel callable from Python via ctypes.

    Keeps the loaded shared library and the constant operator data alive; calling
    the instance applies the kernel to an input vector and returns the output.
    """

    basis: str
    n: int
    _fn: Callable
    _lib: ctypes.CDLL
    _const: dict
    source_path: Path
    lib_path: Path

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_c(np.asarray(x).reshape(self.n), np.float64)
        out = np.empty(self.n, dtype=np.float64)
        if self.basis == "walsh_koopman":
            c = self._const
            self._fn(
                c["rows"].ctypes.data_as(_C_LONG_P),
                c["cols"].ctypes.data_as(_C_LONG_P),
                c["vals"].ctypes.data_as(_C_DOUBLE_P),
                x.ctypes.data_as(_C_DOUBLE_P),
                out.ctypes.data_as(_C_DOUBLE_P),
                ctypes.c_long(c["nnz"]),
                ctypes.c_long(self.n),
            )
        else:  # spectral_svd
            c = self._const
            tmp = np.empty(c["r"], dtype=np.float64)
            self._fn(
                c["U"].ctypes.data_as(_C_DOUBLE_P),
                c["s"].ctypes.data_as(_C_DOUBLE_P),
                c["Vt"].ctypes.data_as(_C_DOUBLE_P),
                x.ctypes.data_as(_C_DOUBLE_P),
                out.ctypes.data_as(_C_DOUBLE_P),
                tmp.ctypes.data_as(_C_DOUBLE_P),
                ctypes.c_long(self.n),
                ctypes.c_long(c["r"]),
            )
        return out


def compile_native(
    result: CompileResult,
    function_name: str = "latent_kernel",
    *,
    out_dir: Path | None = None,
    extra_flags: tuple[str, ...] = ("-O3", "-march=native", "-ffast-math"),
) -> NativeKernel:
    """Compile an accepted latent to machine code and return a callable kernel.

    Raises ``RuntimeError`` if no C compiler is available or compilation fails.
    """
    cc = compiler_available()
    if cc is None:
        raise RuntimeError("no C compiler (clang/gcc) available for native lowering")
    if result.basis not in ("walsh_koopman", "spectral_svd"):
        raise ValueError(f"unknown basis {result.basis!r}")

    src = emit_c_source(result, function_name)
    base = out_dir or Path(tempfile.mkdtemp(prefix="latent_native_"))
    base.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    c_path = base / f"latent_{tag}.c"
    lib_path = base / f"latent_{tag}{_SHARED_EXT}"
    c_path.write_text(src, encoding="utf-8")

    cmd = [cc, *extra_flags, "-shared", "-fPIC", "-o", str(lib_path), str(c_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"native compile failed:\n{proc.stderr}")

    lib = ctypes.CDLL(str(lib_path))
    fn = getattr(lib, function_name)
    fn.restype = None

    rep = result.representation
    if result.basis == "walsh_koopman":
        n = int(rep["shape"][0])
        const = {
            "rows": _as_c(np.asarray(rep["coo_rows"]), np.int64),
            "cols": _as_c(np.asarray(rep["coo_cols"]), np.int64),
            "vals": _as_c(np.asarray(rep["coo_values"]), np.float64),
            "nnz": int(np.asarray(rep["coo_values"]).size),
        }
        fn.argtypes = [_C_LONG_P, _C_LONG_P, _C_DOUBLE_P, _C_DOUBLE_P, _C_DOUBLE_P,
                       ctypes.c_long, ctypes.c_long]
    else:
        n = int(rep["shape"][0])
        const = {
            "U": _as_c(np.asarray(rep["U"]), np.float64),
            "s": _as_c(np.asarray(rep["s"]), np.float64),
            "Vt": _as_c(np.asarray(rep["Vt"]), np.float64),
            "r": int(rep["rank"]),
        }
        fn.argtypes = [_C_DOUBLE_P, _C_DOUBLE_P, _C_DOUBLE_P, _C_DOUBLE_P, _C_DOUBLE_P,
                       _C_DOUBLE_P, ctypes.c_long, ctypes.c_long]

    return NativeKernel(
        basis=result.basis, n=n, _fn=fn, _lib=lib, _const=const,
        source_path=c_path, lib_path=lib_path,
    )
