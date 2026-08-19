"""Latent Compiler — query-only (matrix-free) latent discovery.

The matrix view in ``core.py`` needs the dense operator ``A``: the exact
certificate computes ``||A||_F`` and the truncation error from the full SVD.
Many real operators are never available as a dense matrix — only as a *matvec*
``v -> A v`` (a PDE solve, a graph propagation, an attention map, a simulator
linearization, a kernel applied through a fast transform). For those, discovery
*and* certification must run from matvec queries alone.

This module does exactly that:

  * **Discovery** is a randomized subspace iteration (Halko-Martinsson-Tropp):
    it only ever calls ``matvec`` and (for non-symmetric ``A``) ``rmatvec``, so
    it never forms ``A``.
  * **Certification** is the HMT a-posteriori estimator. The residual operator
    ``R = (I - Q Q^T) A`` is itself matvec-accessible — ``R w = A w - Q (Q^T (A w))``
    — so we probe it with fresh Gaussian vectors and bound its spectral norm:

        P[ ||(I - Q Q^T) A||_2  <=  alpha * max_i ||R w_i||_2 ]  >=  1 - 10^{-p},
        alpha = 10 * sqrt(2/pi),   p = number of probes        (HMT 2011, eq. 4.3).

    This is a *probabilistic* spectral certificate (failure prob 10^{-p}),
    exactly analogous to the probabilistic Boolean-Walsh sketch — and for the
    same reason: without the full operator no procedure can certify exactly.

The cost metric in this regime is the number of matvecs, not flops on a dense
array. Discovery + certification cost O((L + p) + L*n_iter) matvecs, independent
of any dense factorization. Deployment then replaces each expensive black-box
matvec with an O(n r) explicit rank-r factor.

Honesty contract. The certificate is probabilistic and on the *spectral* norm
(not Frobenius); we report the failure probability and the probe count, and the
demo validates the bound on held-out matvecs. A full-rank black box yields a
large residual and is rejected, just like the dense path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# HMT a-posteriori constant: ||(I-QQ^T)A|| <= alpha * max_i ||(I-QQ^T)A w_i||
# with probability >= 1 - 10^{-p} over p Gaussian probes (Halko et al. 2011).
HMT_ALPHA = 10.0 * np.sqrt(2.0 / np.pi)

# A latent is only worth it if its rank is well below the dimension.
DEFAULT_MIN_SPEEDUP = 2.0


@dataclass
class QueryCertificate:
    """Probabilistic spectral certificate for a matrix-free low-rank latent."""

    kind: str  # "probabilistic_spectral"
    spectral_residual_bound: float    # high-prob upper bound on ||A - A_r||_2
    relative_spectral_bound: float    # bound / estimated ||A||_2
    failure_prob: float               # 10^{-p}
    n_probes: int
    tolerance: float
    holds: bool


@dataclass
class QueryCostReport:
    """Cost in matvecs (the real currency when only A@v is available)."""

    discovery_matvecs: int
    certification_matvecs: int
    ambient_matvec_units: float       # cost of one black-box A@v (user-supplied)
    latent_matvec_units: float        # cost of the rank-r factor apply ~ 2 n r
    ideal_speedup: float
    beneficial: bool
    break_even_calls: float | None


@dataclass
class QueryLatentResult:
    accepted: bool
    dimension: int
    rank: int
    tolerance: float
    certificate: QueryCertificate
    cost: QueryCostReport
    reason: str
    representation: dict = field(default_factory=dict)

    def summary(self) -> str:
        verdict = "COMPILED" if self.accepted else "REJECTED"
        return (
            f"[{verdict}] query-only dim={self.dimension} rank={self.rank} "
            f"spec_resid<={self.certificate.spectral_residual_bound:.2e} "
            f"speedup={self.cost.ideal_speedup:.1f}x :: {self.reason}"
        )


def _as_matops(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    rmatvec: Callable[[np.ndarray], np.ndarray] | None,
    symmetric: bool,
    counter: list,
):
    """Wrap matvec/rmatvec to act on matrices (block) and count A-applications."""

    def A(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        cols = M.shape[1] if M.ndim == 2 else 1
        counter[0] += cols
        return matvec(M)

    if symmetric or rmatvec is None:
        At = A
    else:
        def At(M: np.ndarray) -> np.ndarray:
            M = np.asarray(M, dtype=float)
            cols = M.shape[1] if M.ndim == 2 else 1
            counter[0] += cols
            return rmatvec(M)

    return A, At


def query_low_rank_latent(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    tolerance: float = 1e-2,
    sketch_size: int = 32,
    n_iter: int = 4,
    n_probes: int = 6,
    rmatvec: Callable[[np.ndarray], np.ndarray] | None = None,
    symmetric: bool = True,
    ambient_matvec_units: float | None = None,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    seed: int = 0,
) -> QueryLatentResult:
    """Discover and certify a low-rank latent of a black-box operator from matvecs.

    Parameters
    ----------
    matvec:
        ``matvec(V) -> A @ V`` for a matrix ``V`` of shape ``(n, k)`` (and ``(n,)``).
    n:
        Ambient dimension.
    tolerance:
        Target *relative spectral* residual ``||A - A_r||_2 / ||A||_2``.
    sketch_size:
        Range-finder block size ``L`` (the max recoverable rank). Oversample
        beyond the expected rank.
    n_iter:
        Subspace power iterations (sharpens the top subspace; needs ``rmatvec``
        for non-symmetric ``A``).
    n_probes:
        Number of Gaussian probes for the HMT a-posteriori bound. Failure
        probability is ``10^{-n_probes}``.
    rmatvec:
        ``A^T @ V``. Required (for power iteration) when ``symmetric=False``.
    symmetric:
        If ``True``, uses ``matvec`` for ``A^T`` too.
    ambient_matvec_units:
        Cost of one black-box matvec (e.g. ``n*n`` for a dense operator, or a
        large constant for an expensive simulator). Defaults to ``n*n``.
    """
    if sketch_size < 1:
        raise ValueError("sketch_size must be >= 1")
    if not symmetric and rmatvec is None and n_iter > 0:
        raise ValueError("non-symmetric A needs rmatvec for power iteration (or set n_iter=0)")

    rng = np.random.default_rng(seed)
    counter = [0]
    A, At = _as_matops(matvec, n, rmatvec, symmetric, counter)
    L = int(min(sketch_size, n))

    # --- randomized range finding (matvec only) ------------------------------
    Omega = rng.standard_normal((n, L))
    Y = A(Omega)
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(At(Q))
        Q, _ = np.linalg.qr(A(Q))

    # B = Q^T A  (via A^T Q, matvec-only):  B = (A^T Q)^T
    AtQ = At(Q)                                # n x L
    B = AtQ.T                                  # L x n
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub                                 # n x L, approximate left singular vectors
    discovery_matvecs = counter[0]

    # --- HMT a-posteriori spectral residual bound (matvec only) --------------
    cert_counter = [0]
    Ac, _ = _as_matops(matvec, n, rmatvec, symmetric, cert_counter)
    W = rng.standard_normal((n, n_probes))
    AW = Ac(W)
    R = AW - Q @ (Q.T @ AW)                     # (I - QQ^T) A W
    residual_outside_Q = float(HMT_ALPHA * np.max(np.linalg.norm(R, axis=0)))
    certification_matvecs = cert_counter[0]

    s0 = float(s[0]) if s.size and s[0] > 0 else 1.0

    # --- rank selection: keep modes until the spectral residual <= tol*||A|| --
    # Spectral residual of a rank-r truncation: max(s_{r+1}, residual_outside_Q).
    abs_tol = tolerance * s0
    r = L
    for rr in range(1, L + 1):
        s_tail = float(s[rr]) if rr < s.size else 0.0
        if max(s_tail, residual_outside_Q) <= abs_tol:
            r = rr
            break
    spectral_residual_bound = max(float(s[r]) if r < s.size else 0.0, residual_outside_Q)
    relative_spectral_bound = spectral_residual_bound / s0

    holds = relative_spectral_bound <= max(tolerance, 1e-12)
    certificate = QueryCertificate(
        kind="probabilistic_spectral",
        spectral_residual_bound=spectral_residual_bound,
        relative_spectral_bound=relative_spectral_bound,
        failure_prob=10.0 ** (-n_probes),
        n_probes=n_probes,
        tolerance=tolerance,
        holds=holds,
    )

    ambient = float(ambient_matvec_units if ambient_matvec_units is not None else n * n)
    latent = float(max(2 * n * r, 1.0))
    ideal_speedup = ambient / latent
    beneficial = ideal_speedup >= min_speedup
    per_call_gain = ambient - latent
    total_discovery = discovery_matvecs + certification_matvecs
    break_even = (total_discovery * ambient) / per_call_gain if per_call_gain > 0 else None

    cost = QueryCostReport(
        discovery_matvecs=discovery_matvecs,
        certification_matvecs=certification_matvecs,
        ambient_matvec_units=ambient,
        latent_matvec_units=latent,
        ideal_speedup=ideal_speedup,
        beneficial=beneficial,
        break_even_calls=break_even,
    )

    if not holds:
        reason = "no certified low-rank latent from matvec queries at this tolerance"
        accepted = False
    elif not beneficial:
        reason = "certified low-rank latent exists but the rank gives no cost advantage"
        accepted = False
    else:
        reason = "probabilistic-spectral low-rank latent with certified cost advantage"
        accepted = True

    representation = {
        "rank": int(r),
        "U": U[:, :r],
        "s": s[:r],
        "Vt": Vt[:r, :],
        "shape": (n, n),
        "method": "query_only",
        "sketch_size": L,
    }

    return QueryLatentResult(
        accepted=accepted,
        dimension=n,
        rank=int(r),
        tolerance=tolerance,
        certificate=certificate,
        cost=cost,
        reason=reason,
        representation=representation,
    )


def apply_query_latent(result: QueryLatentResult, x: np.ndarray) -> np.ndarray:
    """Apply the discovered rank-r factor A_r = U diag(s) Vt to x (O(n r))."""
    U = np.asarray(result.representation["U"])
    s = np.asarray(result.representation["s"])
    Vt = np.asarray(result.representation["Vt"])
    x = np.asarray(x, dtype=float)
    return U @ (s[:, None] * (Vt @ x)) if x.ndim == 2 else U @ (s * (Vt @ x))
