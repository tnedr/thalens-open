"""
Latent Compiler — Hankel (signal) view: the latent of the latent.

The matrix and Boolean paths in :mod:`latent_compiler.core` ask whether a *given*
operator has a cheap latent. This module asks the recursive question: a slowly
converging series / a black-box trajectory is *itself* a latent representation of
a dynamics (think the Sundman series for the three-body problem). Does that
representation have its own latent?

For a signal that is a finite sum of modes ``x_k = sum_j a_j z_j^k`` the Hankel
matrix ``H[i, j] = x_{i+j}`` has rank equal to the number of modes (Kronecker's
theorem). So:

  * low Hankel rank   <=> few poles / exponential modes <=> the series resums to
    a fast finite-mode model;
  * full Hankel rank  <=> a natural boundary (dense singularities, chaos) <=> no
    cheap latent: certified hardness.

Two further facts make this rigorous and *progressive*:

  * **AAK theory** (Adamyan-Arov-Krein): the optimal rank-``r`` Hankel
    approximation error in the spectral (Hankel) norm is exactly the
    ``(r+1)``-th singular value ``s_r``. So ``s_r / s_0`` is a principled relative
    truncation certificate, not a heuristic.
  * **Weyl + a noise-floor band**: the Hankel is estimated from data with some
    per-sample noise ``eta``. The spectral norm of the noise Hankel is bounded by
    a floor ``nf``; by Weyl's inequality the true singular values lie within
    ``nf`` of the empirical ones. As more samples arrive the *signal* singular
    values grow like ``sqrt(p*q)`` while the *noise* floor grows only like
    ``sqrt(p)+sqrt(q)``, so the relative band shrinks like ``~1/sqrt(N)``. The
    certified rank interval narrows monotonically with data — the same
    certified-anytime behaviour as progressive matrix eigendecomposition
    (PME/CASLA).

This module provides the Hankel construction, the AAK tail certificate, the
noise-floor band, the certified-anytime CASLA decision gate, and matrix-pencil
(ESPRIT-style) mode estimation for actually resumming an accepted series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Numerical floor below which a singular value counts as zero.
_EPS = 1e-12


# --------------------------------------------------------------------------
# Hankel construction and spectrum
# --------------------------------------------------------------------------


def hankel_matrix(x: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    """Build the Hankel matrix ``H[i, j] = x[i + j]`` from a 1D signal.

    A near-square Hankel maximises the signal singular values relative to the
    noise floor, so by default ``n_rows`` is chosen as ``(len(x) + 1) // 2``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 samples to form a Hankel matrix")
    p = (n + 1) // 2 if n_rows is None else int(n_rows)
    p = max(1, min(p, n - 1))
    q = n - p + 1
    idx = np.arange(p)[:, None] + np.arange(q)[None, :]
    return x[idx]


def hankel_singular_values(x: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    """Singular values of the Hankel matrix of ``x`` (descending)."""
    H = hankel_matrix(x, n_rows)
    return np.linalg.svd(H, compute_uv=False)


def aak_relative_tail(s: np.ndarray) -> np.ndarray:
    """Relative AAK truncation certificate per rank.

    ``out[r]`` is ``s[r] / s[0]`` = the exact optimal rank-``r`` Hankel-norm
    approximation error relative to the leading singular value. ``out[0]`` is the
    rank-0 (constant) error = 1 by convention.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s[0] <= _EPS:
        return np.ones(1)
    tail = s / s[0]
    tail[0] = 1.0  # rank-0 (constant) error
    return tail


# --------------------------------------------------------------------------
# Noise-floor band (certified-anytime ingredient)
# --------------------------------------------------------------------------


# Calibrated constant for the analytic Hankel spectral-noise bound. The bound
#
#     ||E||_2 <= C * eta * sqrt( m * ( log(m) + log(K/delta) ) )      (m = max(p,q))
#
# holds with probability >= 1 - delta for an i.i.d. mean-zero, std-eta noise
# Hankel, combining the Meckes mean scaling ``E||E|| ~ sqrt(m log m)`` with a
# sub-Gaussian / Talagrand tail: ``||E||_2`` is ``sqrt(min(p,q))``-Lipschitz in
# the i.i.d. entries, giving the ``log(1/delta)`` term inside the root. ``K`` is
# the number of decision checkpoints, so the union bound makes the band valid
# simultaneously across a progressive schedule (a time-uniform confidence
# sequence). ``C`` is calibrated by Monte-Carlo to dominate the empirical
# (1 - delta) quantile across shapes with margin; the guarantee is regression
# tested in ``test_analytic_floor_dominates_montecarlo``.
HANKEL_NOISE_CONST = 1.3


def analytic_noise_floor(
    shape: tuple[int, int],
    eta: float,
    *,
    delta: float = 0.05,
    n_checkpoints: int = 1,
) -> float:
    """Analytic high-probability / time-uniform bound on ``||noise Hankel||_2``.

    Closed-form replacement for the Monte-Carlo floor: no simulation, ``O(1)``,
    and valid simultaneously across ``n_checkpoints`` budgets (the confidence
    sequence used by the progressive CASLA gate). See ``HANKEL_NOISE_CONST``.
    """
    if eta <= 0.0:
        return 0.0
    m = max(1, max(shape))
    k = max(1, int(n_checkpoints))
    delta = min(max(delta, 1e-12), 1.0)
    return HANKEL_NOISE_CONST * eta * np.sqrt(m * (np.log(m + 1.0) + np.log(k / delta)))


def noise_floor_spectral(
    shape: tuple[int, int],
    eta: float,
    *,
    n_trials: int = 64,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo (1-``quantile``) bound on the spectral norm of a noise Hankel.

    Kept as the empirical reference that calibrates and validates
    :func:`analytic_noise_floor`. ``E`` is a Hankel matrix of i.i.d.
    ``N(0, eta^2)`` samples with the given ``shape``; by Weyl the true singular
    values lie within this floor of the empirical ones.
    """
    if eta <= 0.0:
        return 0.0
    rng = np.random.default_rng() if rng is None else rng
    p, q = shape
    n = p + q - 1
    norms = np.empty(n_trials)
    for t in range(n_trials):
        e = rng.standard_normal(n) * eta
        norms[t] = np.linalg.svd(hankel_matrix(e, n_rows=p), compute_uv=False)[0]
    return float(np.quantile(norms, quantile))


# --------------------------------------------------------------------------
# Interval-threshold decision calculus (shared vocabulary with PME / CASLA)
# --------------------------------------------------------------------------


@dataclass
class IntervalDecision:
    """A three-valued certified decision of ``quantity ? threshold``.

    Mirrors ``spectral_fenton...decide_interval_threshold`` exactly, so the
    Hankel gate is a literal instance of the progressive-matrix-eigendecomposition
    (PME/CASLA) verified anytime decision calculus: given a certified interval
    ``[lower, upper]`` for the true quantity, the decision is sound (never wrong
    while the interval is valid), monotone (a narrower interval never un-decides),
    and stable (once decided it stays decided as the interval shrinks).
    """

    decision: str  # "provably_below" | "provably_above" | "undecided"
    lower: float
    upper: float
    threshold: float
    margin: float


def decide_interval(lower: float, upper: float, threshold: float) -> IntervalDecision:
    """Decide ``quantity <= threshold`` from a certified interval ``[lower, upper]``.

    ``provably_below`` when the whole interval is ``<= threshold``;
    ``provably_above`` when it is ``>= threshold``; ``undecided`` on overlap.
    Identical semantics to the PME decision calculus.
    """
    threshold = float(threshold)
    if upper <= threshold:
        return IntervalDecision("provably_below", lower, upper, threshold, threshold - upper)
    if lower >= threshold:
        return IntervalDecision("provably_above", lower, upper, threshold, lower - threshold)
    return IntervalDecision("undecided", lower, upper, threshold, 0.0)


# --------------------------------------------------------------------------
# CASLA decision gate
# --------------------------------------------------------------------------


@dataclass
class HankelVerdict:
    """Certified-anytime verdict for one sample budget.

    ``rel_tail_upper[r]`` / ``rel_tail_lower[r]`` bracket the *true* relative
    rank-``r`` Hankel tail given the noise floor. ``certified_rank`` is the
    smallest rank whose upper bound meets the tolerance (provably resummable).
    ``rank_exceeds`` is the largest ``r`` whose lower bound still exceeds the
    tolerance (provably *not* rank-``r``: hardness direction).
    """

    n_samples: int
    shape: tuple[int, int]
    sigma1: float
    noise_floor: float
    floor_method: str
    delta: float
    tolerance: float
    max_rank: int
    rel_tail_upper: list[float]
    rel_tail_lower: list[float]
    rank_decisions: list[IntervalDecision]
    certified_rank: int | None
    rank_exceeds: int
    verdict: str  # "resummable" | "no_low_rank_latent" | "undecided"


def hankel_decision(
    x: np.ndarray,
    *,
    tolerance: float,
    eta: float,
    max_rank: int = 16,
    n_rows: int | None = None,
    floor_method: str = "analytic",
    delta: float = 0.05,
    n_checkpoints: int = 1,
    noise_trials: int = 64,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
) -> HankelVerdict:
    """Certified-anytime CASLA gate on the Hankel of a signal.

    For each rank ``r`` a certified interval ``[L_r, U_r]`` brackets the *true*
    relative rank-``r`` Hankel tail (empirical singular value +/- the noise floor,
    via Weyl). The resummability question "is the rank-``r`` tail <= tolerance?"
    is decided by the interval-threshold calculus (:func:`decide_interval`), so
    this gate is a literal instance of the PME/CASLA verified anytime decision
    calculus. Verdicts:

      * ``resummable``         — some rank ``<= max_rank`` is ``provably_below``
        the tolerance: the series resums to a finite-mode model.
      * ``no_low_rank_latent`` — rank ``max_rank`` is ``provably_above`` the
        tolerance: certified hardness up to ``max_rank``.
      * ``undecided``          — the band straddles the tolerance; with more
        samples the analytic floor shrinks (~1/sqrt(N)) and the verdict sharpens.

    ``floor_method="analytic"`` (default) uses the closed-form, time-uniform
    confidence sequence :func:`analytic_noise_floor` (pass ``n_checkpoints`` =
    number of budgets for a progressive schedule). ``"montecarlo"`` uses the
    empirical reference floor.
    """
    H = hankel_matrix(x, n_rows)
    p, q = H.shape
    s = np.linalg.svd(H, compute_uv=False)
    sigma1 = float(s[0]) if s.size else 0.0

    if floor_method == "analytic":
        nf = analytic_noise_floor((p, q), eta, delta=delta, n_checkpoints=n_checkpoints)
    elif floor_method == "montecarlo":
        rng = np.random.default_rng() if rng is None else rng
        nf = noise_floor_spectral((p, q), eta, n_trials=noise_trials, quantile=quantile, rng=rng)
    else:
        raise ValueError(f"floor_method must be 'analytic' or 'montecarlo', got {floor_method!r}")

    R = int(min(max_rank, s.size - 1))
    sigma1_lower = max(sigma1 - nf, _EPS)
    sigma1_upper = sigma1 + nf

    upper: list[float] = []
    lower: list[float] = []
    decisions: list[IntervalDecision] = []
    for r in range(1, R + 1):
        tail_hat = float(s[r]) if r < s.size else 0.0
        u = (tail_hat + nf) / sigma1_lower
        lo = max(0.0, tail_hat - nf) / sigma1_upper
        upper.append(u)
        lower.append(lo)
        decisions.append(decide_interval(lo, u, tolerance))

    certified_rank: int | None = None
    for r, dec in enumerate(decisions, start=1):
        if dec.decision == "provably_below":
            certified_rank = r
            break

    rank_exceeds = 0
    for r, dec in enumerate(decisions, start=1):
        if dec.decision == "provably_above":
            rank_exceeds = r

    if certified_rank is not None:
        verdict = "resummable"
    elif decisions and decisions[-1].decision == "provably_above":
        verdict = "no_low_rank_latent"
    else:
        verdict = "undecided"

    return HankelVerdict(
        n_samples=int(np.asarray(x).size),
        shape=(p, q),
        sigma1=sigma1,
        noise_floor=nf,
        floor_method=floor_method,
        delta=delta,
        tolerance=tolerance,
        max_rank=R,
        rel_tail_upper=upper,
        rel_tail_lower=lower,
        rank_decisions=decisions,
        certified_rank=certified_rank,
        rank_exceeds=rank_exceeds,
        verdict=verdict,
    )


def progressive_hankel_decision(
    x: np.ndarray,
    sample_budgets: list[int],
    *,
    tolerance: float,
    eta: float,
    max_rank: int = 16,
    floor_method: str = "analytic",
    delta: float = 0.05,
    noise_trials: int = 48,
    quantile: float = 0.95,
    seed: int = 0,
) -> list[HankelVerdict]:
    """Run :func:`hankel_decision` at increasing sample budgets.

    Demonstrates the certified-anytime property: as the budget grows the relative
    noise floor shrinks and the verdict sharpens (undecided -> decided), never
    flipping back from a decided state under more data. With the analytic floor
    the band is a *time-uniform* confidence sequence: ``delta`` is split across
    the ``K = len(sample_budgets)`` checkpoints (union bound), so the whole
    trajectory of verdicts holds simultaneously with probability >= 1 - delta.
    """
    rng = np.random.default_rng(seed)
    k = len(sample_budgets)
    out: list[HankelVerdict] = []
    for n in sample_budgets:
        n = int(min(n, np.asarray(x).size))
        out.append(
            hankel_decision(
                np.asarray(x)[:n],
                tolerance=tolerance,
                eta=eta,
                max_rank=max_rank,
                floor_method=floor_method,
                delta=delta,
                n_checkpoints=k,
                noise_trials=noise_trials,
                quantile=quantile,
                rng=rng,
            )
        )
    return out


# --------------------------------------------------------------------------
# Resummation: matrix-pencil (ESPRIT) mode estimation
# --------------------------------------------------------------------------


@dataclass
class ModeModel:
    """A finite-mode model ``x_k ~= sum_j amplitudes_j * poles_j^k``."""

    poles: np.ndarray  # complex, length r
    amplitudes: np.ndarray  # complex, length r
    rank: int
    fit_rel_error: float = 0.0
    representation: dict = field(default_factory=dict)

    def predict(self, indices: np.ndarray) -> np.ndarray:
        """Evaluate the model at arbitrary integer indices (real part)."""
        k = np.asarray(indices, dtype=float)
        # Stable evaluation: poles^k via exp(k * log(pole)) with magnitude clip
        # only matters for |pole|>1; resummable signals have |pole|~1.
        vand = self.poles[None, :] ** k[:, None]
        return np.real(vand @ self.amplitudes)


def estimate_modes(x: np.ndarray, rank: int, n_rows: int | None = None) -> ModeModel:
    """Matrix-pencil / ESPRIT estimation of the dominant ``rank`` signal modes.

    The signal subspace is the leading ``rank`` left singular vectors of the
    Hankel matrix; the per-step poles are the eigenvalues of the shift operator
    on that subspace; the amplitudes are recovered by a Vandermonde least-squares
    fit. This is how an accepted low-rank Hankel latent becomes a fast evaluator
    that resums the original slow series.
    """
    x = np.asarray(x, dtype=float).ravel()
    H = hankel_matrix(x, n_rows)
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    r = int(max(1, min(rank, U.shape[1], H.shape[0] - 1)))
    Ur = U[:, :r]
    up = Ur[:-1]
    dn = Ur[1:]
    shift, *_ = np.linalg.lstsq(up, dn, rcond=None)
    poles = np.linalg.eigvals(shift)

    k = np.arange(x.size)
    vand = poles[None, :] ** k[:, None]
    amps, *_ = np.linalg.lstsq(vand, x.astype(complex), rcond=None)
    fit = np.real(vand @ amps)
    denom = float(np.linalg.norm(x))
    fit_err = float(np.linalg.norm(x - fit) / denom) if denom > _EPS else 0.0

    return ModeModel(
        poles=poles,
        amplitudes=amps,
        rank=r,
        fit_rel_error=fit_err,
        representation={"singular_values": s[: r + 1].tolist()},
    )


def resummation_extrapolation_error(
    x: np.ndarray,
    rank: int,
    fit_fraction: float = 0.5,
) -> dict:
    """Honest resummation test: fit modes on a prefix, predict the held-out tail.

    A genuine latent of the series must *extrapolate*, not just interpolate. We
    fit the mode model on the first ``fit_fraction`` of the samples and measure
    the relative error on the unseen remainder. Low held-out error = the slow
    series has been resummed into a fast finite-mode evaluator.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    cut = int(max(rank * 2 + 2, fit_fraction * n))
    cut = min(cut, n - 1)
    model = estimate_modes(x[:cut], rank)
    pred_tail = model.predict(np.arange(cut, n))
    true_tail = x[cut:]
    denom = float(np.linalg.norm(true_tail))
    held_out = float(np.linalg.norm(true_tail - pred_tail) / denom) if denom > _EPS else 0.0
    return {
        "rank": model.rank,
        "fit_samples": cut,
        "held_out_samples": n - cut,
        "fit_rel_error": model.fit_rel_error,
        "held_out_rel_error": held_out,
        "max_pole_magnitude": float(np.max(np.abs(model.poles))),
    }
