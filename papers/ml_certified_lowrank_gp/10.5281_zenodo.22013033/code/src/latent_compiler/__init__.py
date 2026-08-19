"""
Latent Compiler
===============

Find the certified, cost-beneficial latent of a computation — or prove there is
none.

A computation is supplied as a finite Boolean map (machine-level view) or as a
linear operator (matrix view). The compiler extracts it as an operator in a
natural spectral basis (Walsh-Koopman for Boolean, SVD for linear), discovers a
sparse / low-rank latent, certifies exact or bounded-error equivalence, accounts
the cost against ambient execution, and accepts the latent only when it is both
certified and cheaper. Accepted latents lower to a runnable NumPy kernel.

Usage:
    >>> import numpy as np
    >>> import latent_compiler as lc
    >>> A = np.outer(np.arange(64.0), np.arange(64.0))  # rank-1 operator
    >>> result = lc.latent_compile(A, tolerance=1e-9)
    >>> result.accepted, result.representation["rank"]
    (True, 1)

    >>> rot = lambda x, m=10: ((x << 1) | (x >> (m - 1))) & ((1 << m) - 1)
    >>> r = lc.latent_compile(lambda x: rot(x), m_bits=10, vectorized=False)
    >>> r.accepted  # rotation has a one-term-per-column Walsh-Koopman latent
    True
"""

from .core import (
    Certificate,
    CompileResult,
    CostReport,
    compile_boolean,
    compile_matrix,
    hadamard_sign_matrix,
    km_heavy_hitters,
    latent_compile,
    normalized_latent_complexity,
    participation_ratio,
    randomized_svd,
    sketched_boolean_latent,
    sketched_low_rank_latent,
    walsh_koopman_operator,
)
from .hankel import (
    HankelVerdict,
    IntervalDecision,
    ModeModel,
    aak_relative_tail,
    analytic_noise_floor,
    decide_interval,
    estimate_modes,
    hankel_decision,
    hankel_matrix,
    hankel_singular_values,
    noise_floor_spectral,
    progressive_hankel_decision,
    resummation_extrapolation_error,
)
from .lowering import build_executor, emit_source, round_trip_error
from .native import (
    NativeKernel,
    compile_native,
    compiler_available,
    emit_c_source,
)
from .operator import (
    QueryCertificate,
    QueryCostReport,
    QueryLatentResult,
    apply_query_latent,
    query_low_rank_latent,
)
from .resolvent import (
    CertifiedResolvent,
    certify_resolvent,
    resolvent_solve,
)

__all__ = [
    "Certificate",
    "CompileResult",
    "CostReport",
    "latent_compile",
    "compile_boolean",
    "compile_matrix",
    "walsh_koopman_operator",
    "hadamard_sign_matrix",
    "participation_ratio",
    "normalized_latent_complexity",
    "randomized_svd",
    "sketched_low_rank_latent",
    "km_heavy_hitters",
    "sketched_boolean_latent",
    "hankel_matrix",
    "hankel_singular_values",
    "aak_relative_tail",
    "noise_floor_spectral",
    "analytic_noise_floor",
    "IntervalDecision",
    "decide_interval",
    "HankelVerdict",
    "hankel_decision",
    "progressive_hankel_decision",
    "ModeModel",
    "estimate_modes",
    "resummation_extrapolation_error",
    "build_executor",
    "emit_source",
    "round_trip_error",
    "emit_c_source",
    "compile_native",
    "compiler_available",
    "NativeKernel",
    "CertifiedResolvent",
    "certify_resolvent",
    "resolvent_solve",
    "QueryCertificate",
    "QueryCostReport",
    "QueryLatentResult",
    "query_low_rank_latent",
    "apply_query_latent",
]

__version__ = "0.1.0"
__author__ = "Dr. Tamás Nagy"
