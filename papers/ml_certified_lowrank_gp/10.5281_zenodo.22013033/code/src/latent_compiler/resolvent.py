"""Latent Compiler — the noise-floor rule for regularized resolvent solves.

This module promotes the rule discovered in the certified Gaussian-process work
into a *general* certified-substitution principle. Many widely used computations
end in a regularized resolvent solve

    x = (A + delta * I)^{-1} b ,      A symmetric PSD,  delta > 0,

where ``delta`` is a known positive floor: observation-noise variance in a
Gaussian process, the ridge parameter in kernel ridge regression, the Tikhonov
parameter in a linear inverse problem, the damping term in a Levenberg-Marquardt
/ damped-Newton step, or the teleport mass in a regularized graph diffusion.

The temptation is to replace the expensive operator ``A`` by a cheap low-rank
surrogate ``A_r`` and solve with it. The danger is that the *operator* error
``||A - A_r||`` is amplified by the resolvent: a 1% operator error with a tiny
floor can produce an order-of-magnitude error in ``x``. So certifying the
operator is not enough; we must certify the *output*.

The noise-floor rule (one-line theorem)
----------------------------------------
Let ``E = A - A_r`` with ``A_r`` symmetric PSD (so ``A_r + delta I >= delta I``).
With ``A = A_full + delta I``, ``A_hat = A_r + delta I``, ``x = A^{-1} b``,
``x_hat = A_hat^{-1} b``:

    x_hat - x = A_hat^{-1} E A^{-1} b = A_hat^{-1} E x
    => ||x_hat - x|| / ||x|| <= ||A_hat^{-1}|| * ||E|| <= ||E||_2 / delta
                                                       <= ||E||_F / delta.

Therefore, if we pick the rank so that

    ||A - A_r||_F <= eps * delta          (the noise-floor rule)

then the relative output error of the resolvent solve is certified ``<= eps``,
independent of how ill-conditioned ``A`` itself is. The floor ``delta`` is the
amplification budget; the rule spends it deliberately.

This is the unifying statement behind the GP result: GP, kernel ridge, Tikhonov,
damped Newton, and regularized graph diffusion are all the same resolvent, and
the same single rule certifies a low-rank substitution for every one of them.

What this module does NOT claim
-------------------------------
* It is a *sufficient* certificate, not a tight one: the spectral norm of ``E``
  may be far below ``||E||_F``, so the true error is often much smaller than the
  bound (the demos show this).
* It requires ``A`` symmetric PSD and an explicit positive floor. General
  (non-PSD, unregularized) solves are out of scope — there the resolvent has no
  floor and no finite amplification budget.
* The factor is discovered by the latent compiler (exact or sketched SVD); this
  module owns only the noise-floor tolerance choice, the PSD-safe Woodbury solve,
  and the propagated output certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import latent_compile


@dataclass
class CertifiedResolvent:
    """A certified low-rank substitution for ``(A + delta I)^{-1}``.

    Attributes
    ----------
    accepted:
        Whether the latent compiler accepted the low-rank factor (certified at
        the noise floor *and* cost-beneficial). If ``False``, fall back to the
        exact solve; the surrogate is not safe/worthwhile.
    rank:
        Rank of the discovered PSD factor ``A_r = W W^T``.
    floor:
        The resolvent floor ``delta`` (regularizer / noise term).
    operator_error_fro:
        Certified ``||A - A_r||_F`` (absolute Frobenius).
    rel_output_bound:
        Certified relative error of the solve ``x_hat`` vs the exact ``x``:
        ``||x - x_hat|| / ||x|| <= operator_error_fro / floor``.
    eps_target:
        The accuracy target the rank was chosen for (``operator_error <= eps*floor``).
    reason:
        Human-readable verdict from the underlying compile.
    """

    accepted: bool
    rank: int
    floor: float
    operator_error_fro: float
    rel_output_bound: float
    eps_target: float
    reason: str
    factor: np.ndarray | None  # W with A_r = W W^T, shape (n, rank)


def certify_resolvent(
    A: np.ndarray,
    floor: float,
    eps: float,
    *,
    method: str = "sketched",
    max_rank: int | None = None,
    min_speedup: float = 2.0,
    seed: int = 0,
) -> CertifiedResolvent:
    """Discover a low-rank PSD factor for ``A`` obeying the noise-floor rule.

    Picks the rank so that ``||A - A_r||_F <= eps * floor`` and returns the PSD
    factor ``W`` (``A_r = W W^T``) together with the certified relative output
    bound ``eps`` for the resolvent solve ``(A + floor I)^{-1} b``.

    Parameters
    ----------
    A:
        Symmetric PSD operator (kernel/Gram/Laplacian-shifted matrix).
    floor:
        Positive resolvent floor ``delta`` (noise variance, ridge, Tikhonov, ...).
    eps:
        Target relative output accuracy.
    method:
        ``"sketched"`` (randomized range finder, scalable) or ``"exact"`` (full SVD).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    if floor <= 0.0:
        raise ValueError("floor (delta) must be strictly positive — the resolvent needs a budget")
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive")

    Afro = float(np.linalg.norm(A, "fro"))
    if Afro <= 0.0:
        return CertifiedResolvent(True, 0, floor, 0.0, 0.0, eps, "zero operator", None)

    # The noise-floor rule, expressed as the compiler's relative-Frobenius tol.
    tol_rel = eps * floor / Afro

    if method == "sketched":
        res = latent_compile(
            A, tolerance=tol_rel, value_width=1, method="sketched",
            max_rank=max_rank, min_speedup=min_speedup, seed=seed,
        )
    else:
        res = latent_compile(A, tolerance=tol_rel, value_width=1, min_speedup=min_speedup)

    operator_error = float(res.certificate.relative_error) * Afro
    rel_bound = operator_error / floor

    if not res.accepted:
        return CertifiedResolvent(
            accepted=False,
            rank=int(res.representation.get("rank", 0)),
            floor=floor,
            operator_error_fro=operator_error,
            rel_output_bound=rel_bound,
            eps_target=eps,
            reason=res.reason,
            factor=None,
        )

    # PSD factor: truncated SVD of a symmetric PSD matrix gives A_r = U diag(s) U^T
    # (U == V up to sign for PSD); W = U sqrt(s) so A_r = W W^T.
    U = np.asarray(res.representation["U"])
    s = np.asarray(res.representation["s"])
    W = U * np.sqrt(np.maximum(s, 0.0))[None, :]

    return CertifiedResolvent(
        accepted=True,
        rank=int(res.representation["rank"]),
        floor=floor,
        operator_error_fro=operator_error,
        rel_output_bound=rel_bound,
        eps_target=eps,
        reason=res.reason,
        factor=W,
    )


def resolvent_solve(W: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    """Solve ``(W W^T + floor I) x = b`` in O(n r^2 + r^3) via Woodbury.

    ``(floor I + W W^T)^{-1} = floor^{-1} I - floor^{-2} W (I_r + floor^{-1} W^T W)^{-1} W^T``.
    """
    n, r = W.shape
    b = np.asarray(b, dtype=float)
    WtW = W.T @ W
    M = np.eye(r) + WtW / floor
    Wtb = W.T @ b
    correction = W @ np.linalg.solve(M, Wtb)
    return b / floor - correction / (floor * floor)
