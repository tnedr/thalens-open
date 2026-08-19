"""
Latent Compiler — lowering and round-trip verification.

Once a latent is accepted, it must become something executable. This module
turns a ``CompileResult`` into:

  - a runnable executor (a Python closure backed by NumPy), and
  - emitted self-contained source code for that executor,

and it provides a round-trip check that the cheap latent executor agrees with
the dense ambient operator within the certified tolerance.

The lowered artifact propagates *observables*:

  - Boolean (Walsh-Koopman): given an observable expressed in Walsh
    coefficients ``g``, the executor returns the coefficients of ``g o F`` via
    the sparse operator ``K_sparse @ g``.
  - Matrix (spectral SVD): given a value matrix ``V``, the executor returns
    ``A V`` via the low-rank product ``U (s * (Vt V))``.

This is a proof-of-concept lowering to NumPy. Lowering further to SIMD/GPU/FPGA
is downstream and out of scope here; the contract (sparse/low-rank kernel plus
certificate) is what a hardware backend would consume.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .core import CompileResult


def build_executor(result: CompileResult) -> Callable[[np.ndarray], np.ndarray]:
    """Return a runnable latent executor for an accepted compile result."""
    if result.basis == "walsh_koopman":
        rep = result.representation
        n = rep["shape"][0]
        rows = np.asarray(rep["coo_rows"])
        cols = np.asarray(rep["coo_cols"])
        vals = np.asarray(rep["coo_values"])

        def boolean_executor(g: np.ndarray) -> np.ndarray:
            g = np.asarray(g, dtype=float).reshape(n)
            out = np.zeros(n, dtype=float)
            np.add.at(out, rows, vals * g[cols])
            return out

        return boolean_executor

    if result.basis == "spectral_svd":
        rep = result.representation
        U = np.asarray(rep["U"])
        s = np.asarray(rep["s"])
        Vt = np.asarray(rep["Vt"])

        def matrix_executor(V: np.ndarray) -> np.ndarray:
            V = np.asarray(V, dtype=float)
            return U @ (s[:, None] * (Vt @ V))

        return matrix_executor

    raise ValueError(f"unknown basis {result.basis!r}")


def emit_source(result: CompileResult, function_name: str = "latent_kernel") -> str:
    """Emit self-contained NumPy source code implementing the latent executor."""
    if result.basis == "walsh_koopman":
        rep = result.representation
        n = rep["shape"][0]
        return (
            "import numpy as np\n\n"
            f"# Sparse Walsh-Koopman latent kernel ({rep['nnz']} nonzeros, dim {n}).\n"
            f"# Propagates an observable's Walsh coefficients through F.\n"
            "_ROWS = np.array(_rows)\n"
            "_COLS = np.array(_cols)\n"
            "_VALS = np.array(_vals)\n\n"
            f"def {function_name}(g):\n"
            f"    g = np.asarray(g, dtype=float).reshape({n})\n"
            f"    out = np.zeros({n}, dtype=float)\n"
            "    np.add.at(out, _ROWS, _VALS * g[_COLS])\n"
            "    return out\n"
        )

    if result.basis == "spectral_svd":
        rep = result.representation
        r = rep["rank"]
        n = rep["shape"][0]
        return (
            "import numpy as np\n\n"
            f"# Low-rank spectral latent kernel (rank {r}, dim {n}).\n"
            "# Computes A @ V via U (s * (Vt @ V)).\n"
            "_U = np.array(_U_data)\n"
            "_S = np.array(_s_data)\n"
            "_VT = np.array(_Vt_data)\n\n"
            f"def {function_name}(V):\n"
            "    V = np.asarray(V, dtype=float)\n"
            "    return _U @ (_S[:, None] * (_VT @ V))\n"
        )

    raise ValueError(f"unknown basis {result.basis!r}")


def round_trip_error(
    result: CompileResult,
    dense_operator: np.ndarray,
    trials: int = 8,
    seed: int = 0,
) -> float:
    """Max relative error between the latent executor and the dense operator.

    For the Boolean basis, ``dense_operator`` is the full Walsh-Koopman matrix K
    and we compare ``K_sparse @ g`` against ``K @ g`` on random observables.
    For the matrix basis, ``dense_operator`` is A and we compare the low-rank
    product against ``A @ V`` on random value matrices.
    """
    rng = np.random.default_rng(seed)
    executor = build_executor(result)
    dense_operator = np.asarray(dense_operator, dtype=float)
    n = dense_operator.shape[0]

    worst = 0.0
    width = result.representation.get("value_width", 1) if result.basis == "spectral_svd" else 1
    for _ in range(trials):
        V = rng.standard_normal((n, width)) if width > 1 else rng.standard_normal(n)
        reference = dense_operator @ V
        produced = executor(V)
        denom = float(np.linalg.norm(reference))
        if denom <= 1e-12:
            continue
        worst = max(worst, float(np.linalg.norm(reference - produced) / denom))
    return worst
