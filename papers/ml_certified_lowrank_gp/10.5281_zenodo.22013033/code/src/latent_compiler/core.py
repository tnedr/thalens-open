"""
Latent Compiler — core pipeline.

The latent compiler answers one question about a computation:

    Is there a basis in which this computation is cheap, and can we certify it?

A computation is supplied either as a finite Boolean map F: {0,1}^m -> {0,1}^m
(the machine-level view) or as a linear operator A on R^n (the matrix view).
The compiler:

  1. extracts the computation as an operator in a natural spectral basis,
  2. discovers a sparse / low-rank latent representation,
  3. certifies exact or bounded-error equivalence,
  4. accounts the cost against ambient execution,
  5. accepts the latent only when it is both certified and cost-beneficial.

This module implements steps 1-5 and the result types. Lowering the accepted
latent to a runnable executor / source lives in ``lowering.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# A latent that is certified but only marginally cheaper is not worth it.
# A replacement is "beneficial" only above this ideal speedup.
DEFAULT_MIN_SPEEDUP = 2.0

# Numerical floor below which a spectral coefficient counts as zero.
COEFF_EPS = 1e-12

# Guard: the dense Boolean Koopman operator is 2^m x 2^m. Above this it is
# refused rather than silently exhausting memory. Larger maps need the sampled
# discovery path (future work), so we fail loudly instead of pretending.
MAX_BOOLEAN_BITS = 14


@dataclass
class Certificate:
    """Evidence that the latent representation equals (or approximates) the original."""

    kind: str  # "exact" | "bounded" | "none"
    relative_error: float
    tolerance: float
    verified_on: str  # "full_state_space" | "operator_norm"
    holds: bool


@dataclass
class CostReport:
    """Ambient vs latent execution cost, with the amortization break-even point."""

    ambient_units: float
    latent_units: float
    discovery_units: float
    ideal_speedup: float
    beneficial: bool
    break_even_calls: float | None


@dataclass
class CompileResult:
    """The full outcome of one compile attempt."""

    accepted: bool
    basis: str  # "walsh_koopman" | "spectral_svd"
    dimension: int
    tolerance: float
    latent_complexity: float
    certificate: Certificate
    cost: CostReport
    reason: str
    representation: dict = field(default_factory=dict)

    def summary(self) -> str:
        verdict = "COMPILED" if self.accepted else "REJECTED"
        return (
            f"[{verdict}] basis={self.basis} dim={self.dimension} "
            f"err={self.certificate.relative_error:.2e} "
            f"speedup={self.cost.ideal_speedup:.1f}x "
            f"latent_complexity={self.latent_complexity:.4f} :: {self.reason}"
        )


# --------------------------------------------------------------------------
# Step 1: operator extraction
# --------------------------------------------------------------------------


def _parity_popcount(values: np.ndarray, m_bits: int) -> np.ndarray:
    acc = np.zeros_like(values)
    for bit in range(m_bits):
        acc ^= (values >> bit) & 1
    return acc


def hadamard_sign_matrix(m_bits: int) -> np.ndarray:
    """H[T, x] = (-1)^{popcount(T & x)} over {0,1}^m (the Walsh character table)."""
    n = 1 << m_bits
    idx = np.arange(n, dtype=np.int64)
    both = idx[:, None] & idx[None, :]
    return np.where(_parity_popcount(both, m_bits) == 0, 1.0, -1.0)


def boolean_truth_table(F: Callable[[int], int], m_bits: int, vectorized: bool = False) -> np.ndarray:
    """Evaluate F on the whole state space, returning F(x) for x = 0..2^m-1."""
    n = 1 << m_bits
    states = np.arange(n, dtype=np.int64)
    if vectorized:
        values = np.asarray(F(states), dtype=np.int64)
    else:
        values = np.fromiter((int(F(int(x))) for x in states), dtype=np.int64, count=n)
    return values & (n - 1)


def walsh_koopman_operator(truth_table: np.ndarray, m_bits: int) -> np.ndarray:
    """Koopman operator of a Boolean map in the Walsh basis.

    Column S is the Walsh expansion of chi_S(F(x)); K = (1/n) H @ H[F(x)].
    """
    n = 1 << m_bits
    if truth_table.shape != (n,):
        raise ValueError(f"truth_table must have shape ({n},), got {truth_table.shape}")
    H = hadamard_sign_matrix(m_bits)
    V = H[truth_table]
    return (H @ V) / n


# --------------------------------------------------------------------------
# Step 2: latent discovery
# --------------------------------------------------------------------------


def participation_ratio(columns: np.ndarray) -> np.ndarray:
    """Per-column effective number of active coefficients: (sum c^2)^2 / sum c^4."""
    sq = columns**2
    num = sq.sum(axis=0) ** 2
    den = (sq**2).sum(axis=0)
    den = np.where(den <= 0.0, 1.0, den)
    return num / den


def normalized_latent_complexity(operator: np.ndarray) -> float:
    """Mean participation ratio mapped to [0, 1]: 0 = perfect latent, ~1 = chaotic."""
    n = operator.shape[0]
    if n <= 1:
        return 0.0
    mean_pr = float(participation_ratio(operator).mean())
    return (mean_pr - 1.0) / (n - 1.0)


def sparse_column_latent(K: np.ndarray, tolerance: float) -> np.ndarray:
    """Keep the largest coefficients per column until residual energy <= tolerance^2."""
    n = K.shape[1]
    selected = np.zeros_like(K, dtype=bool)
    target_energy = max(0.0, min(1.0, 1.0 - tolerance**2))
    for col in range(n):
        energy = K[:, col] ** 2
        total = float(energy.sum())
        if total <= COEFF_EPS:
            continue
        if tolerance <= 0.0:
            selected[:, col] = np.abs(K[:, col]) > COEFF_EPS
            continue
        order = np.argsort(-energy)
        cumulative = np.cumsum(energy[order]) / total
        keep = int(np.searchsorted(cumulative, target_energy, side="left") + 1)
        selected[order[:keep], col] = True
    return np.where(selected, K, 0.0)


def _rank_for_tail(s: np.ndarray, tolerance: float) -> int:
    """Smallest rank whose relative Frobenius tail energy is within tolerance."""
    if tolerance <= 0.0:
        return max(1, int(np.sum(s > COEFF_EPS * s[0])))
    # Eckart-Young: ||A - A_r||_F / ||A||_F = sqrt(sum_{i>=r} s_i^2 / sum_i s_i^2).
    energy = s**2
    total = float(energy.sum())
    tail = total - np.cumsum(energy)
    rel_tail = np.sqrt(np.maximum(tail, 0.0) / total)
    within = np.where(rel_tail <= tolerance)[0]
    return int(within[0] + 1) if within.size else len(s)


def low_rank_latent(A: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Truncated SVD keeping the smallest rank with relative Frobenius error <= tolerance."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if float(np.linalg.norm(s)) <= COEFF_EPS:
        return U[:, :1], s[:1], Vt[:1, :], 1
    r = max(1, _rank_for_tail(s, tolerance))
    return U[:, :r], s[:r], Vt[:r, :], r


def randomized_svd(
    A: np.ndarray, sketch_size: int, n_iter: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized range-finder SVD (Halko-Martinsson-Tropp) keeping ``sketch_size`` modes.

    Instead of factoring the whole operator, probe its action on a few random
    directions, orthonormalize the response (the *sketch*), and factor the small
    projected operator. This is the engine of sketched discovery.
    """
    m = A.shape[1]
    L = int(min(max(1, sketch_size), m))
    omega = rng.standard_normal((m, L))
    Q, _ = np.linalg.qr(A @ omega)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub, s, Vt


def sketched_low_rank_latent(
    A: np.ndarray,
    tolerance: float,
    max_rank: int | None = None,
    oversample: int = 10,
    n_iter: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """Discover a low-rank latent by sketching, without a full SVD.

    Returns ``(U, s, Vt, r, relative_error, discovery_columns)``. The relative
    error is *exact* for the produced rank-r latent: given the sketch basis Q,
    ``||A - A_r||_F^2 = (||A||_F^2 - ||Q^T A||_F^2) + sum_{i>=r} s_i^2``, so the
    certificate stays rigorous even though discovery only touched a few probes.
    The sketch size doubles until the achievable error meets the tolerance or the
    sketch spans the whole space (which recovers the exact factorization).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = A.shape[0]
    total_energy = float(np.sum(A * A))
    if total_energy <= COEFF_EPS:
        z = np.zeros((n, 1))
        return z, np.zeros(1), np.zeros((1, n)), 1, 0.0, 0.0

    base = max_rank if max_rank else 16
    L = int(min(n, base + oversample))
    discovery_columns = 0.0
    U = s = Vt = None
    relative_error = 1.0
    r = 1
    while True:
        U, s, Vt = randomized_svd(A, L, n_iter, rng)
        discovery_columns += L * (2 * n_iter + 2)
        outside = max(0.0, total_energy - float(np.sum(s**2)))
        r = max(1, _rank_for_tail(s, tolerance))
        captured_tail = max(0.0, float(np.sum(s[r:] ** 2)))
        relative_error = float(np.sqrt((outside + captured_tail) / total_energy))
        achievable = float(np.sqrt(outside / total_energy))
        meets = relative_error <= max(tolerance, 1e-10)
        if meets or L >= n or achievable <= max(tolerance, 1e-10):
            break
        L = int(min(n, L * 2))

    return U[:, :r], s[:r], Vt[:r, :], r, relative_error, discovery_columns


# Guard for the sketched Boolean path: the lowered executor allocates a vector
# of length 2^m, so we cap m to keep memory bounded while still going far beyond
# the exact-enumeration limit (2^22 ~ 4M floats ~ 32 MB).
MAX_BOOLEAN_SKETCH_BITS = 22


def _boolean_pm_observable(
    F: Callable[[np.ndarray], np.ndarray], S: int, m_bits: int, vectorized: bool
) -> Callable[[np.ndarray], np.ndarray]:
    """Return g_S(x) = (-1)^<S, F(x)> as a +/-1 valued function on {0,1}^m."""
    mask = (1 << m_bits) - 1

    def g(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int64)
        if vectorized:
            fx = np.asarray(F(x), dtype=np.int64) & mask
        else:
            fx = np.fromiter((int(F(int(v))) & mask for v in x), dtype=np.int64, count=x.size)
        return 1.0 - 2.0 * _parity_popcount(fx & S, m_bits).astype(np.float64)

    return g


def _km_bucket_energy(
    g: Callable[[np.ndarray], np.ndarray],
    m_bits: int,
    k: int,
    prefix: int,
    n_samples: int,
    rng: np.random.Generator,
    counter: list,
) -> float:
    """Unbiased estimate of the Walsh energy in the bucket of frequencies whose
    low k bits equal ``prefix``: W(k, prefix) = sum_{T: T[:k]=prefix} g_hat(T)^2.

    Uses the paired-sample estimator
        W = E[ g(z, y) g(z', y) (-1)^<prefix, z ^ z'> ]
    over y in {0,1}^{m-k} (shared high bits) and independent z, z' in {0,1}^k.
    """
    if k == 0:
        return 1.0
    n_high = m_bits - k
    if n_high > 0:
        high = rng.integers(0, 1 << n_high, size=n_samples).astype(np.int64) << k
    else:
        high = np.zeros(n_samples, dtype=np.int64)
    za = rng.integers(0, 1 << k, size=n_samples).astype(np.int64)
    zb = rng.integers(0, 1 << k, size=n_samples).astype(np.int64)
    ga = g(high | za)
    gb = g(high | zb)
    counter[0] += 2 * n_samples
    sign = 1.0 - 2.0 * _parity_popcount(prefix & (za ^ zb), k).astype(np.float64)
    return float(np.mean(ga * gb * sign))


def _km_coefficient(
    g: Callable[[np.ndarray], np.ndarray],
    m_bits: int,
    T: int,
    n_samples: int,
    rng: np.random.Generator,
    counter: list,
) -> float:
    """Monte-Carlo estimate of a single Walsh coefficient g_hat(T) = E_x[g(x)(-1)^<T,x>]."""
    x = rng.integers(0, 1 << m_bits, size=n_samples).astype(np.int64)
    gv = g(x)
    counter[0] += n_samples
    sign = 1.0 - 2.0 * _parity_popcount(T & x, m_bits).astype(np.float64)
    return float(np.mean(gv * sign))


def km_heavy_hitters(
    g: Callable[[np.ndarray], np.ndarray],
    m_bits: int,
    threshold: float,
    n_samples: int,
    rng: np.random.Generator,
    counter: list,
    max_active: int = 256,
) -> dict:
    """Kushilevitz-Mansour search for all Walsh frequencies with energy >= threshold.

    Splits the frequency space one bit at a time, estimates each bucket's energy
    by sampling, and prunes buckets below threshold. Survivors at depth m are the
    heavy frequencies; their coefficients are estimated directly. Cost is
    poly(m, 1/threshold), independent of the 2^m operator size.
    """
    prefixes = [0]
    for k in range(m_bits):
        scored = []
        for p in prefixes:
            for bit in (0, 1):
                child = p | (bit << k)
                w = _km_bucket_energy(g, m_bits, k + 1, child, n_samples, rng, counter)
                if w >= 0.5 * threshold:
                    scored.append((w, child))
        if not scored:
            return {}
        scored.sort(reverse=True)
        prefixes = [c for _, c in scored[:max_active]]

    coeffs: dict = {}
    for T in prefixes:
        c = _km_coefficient(g, m_bits, T, 4 * n_samples, rng, counter)
        if c * c >= 0.5 * threshold:
            coeffs[int(T)] = c
    return coeffs


def sketched_boolean_latent(
    F: Callable[[np.ndarray], np.ndarray],
    m_bits: int,
    observables: list,
    tolerance: float,
    n_samples: int,
    threshold: float,
    vectorized: bool,
    rng: np.random.Generator,
) -> tuple[list, list, list, dict, float, int]:
    """Discover sparse Walsh-Koopman columns for a set of observables by sampling.

    For each observable S, ``g_S`` is a +/-1 function with total Walsh energy 1
    (Parseval). KM finds its heavy coefficients; we keep the largest until the
    captured energy reaches 1 - tolerance^2. The residual per column is therefore
    1 - (captured energy), a probabilistic certificate estimated from samples.

    Returns ``(rows, cols, vals, per_column_residual, relative_error, n_evals)``.
    """
    counter = [0]
    rows: list = []
    cols: list = []
    vals: list = []
    per_column_residual: dict = {}
    captured_total = 0.0

    for S in observables:
        g = _boolean_pm_observable(F, int(S), m_bits, vectorized)
        coeffs = km_heavy_hitters(g, m_bits, threshold, n_samples, rng, counter)
        items = sorted(coeffs.items(), key=lambda kv: kv[1] ** 2, reverse=True)
        captured = 0.0
        for T, c in items:
            rows.append(int(T))
            cols.append(int(S))
            vals.append(float(c))
            captured += c * c
            if captured >= 1.0 - tolerance**2:
                break
        per_column_residual[int(S)] = max(0.0, 1.0 - captured)
        captured_total += min(1.0, captured)

    n_cols = max(1, len(observables))
    relative_error = float(np.sqrt(max(0.0, 1.0 - captured_total / n_cols)))
    return rows, cols, vals, per_column_residual, relative_error, counter[0]


# --------------------------------------------------------------------------
# Step 3 + 4: certification and cost accounting
# --------------------------------------------------------------------------


def _relative_frobenius(reference: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    if denom <= COEFF_EPS:
        return 0.0
    return float(np.linalg.norm(reference - approx) / denom)


def _certificate(relative_error: float, tolerance: float, verified_on: str) -> Certificate:
    holds = relative_error <= max(tolerance, 1e-10)
    if not holds:
        kind = "none"
    elif relative_error <= 1e-10:
        kind = "exact"
    else:
        kind = "bounded"
    return Certificate(
        kind=kind,
        relative_error=relative_error,
        tolerance=tolerance,
        verified_on=verified_on,
        holds=holds,
    )


def _cost_report(
    ambient_units: float,
    latent_units: float,
    discovery_units: float,
    min_speedup: float,
) -> CostReport:
    latent_units = max(latent_units, 1.0)
    ideal_speedup = ambient_units / latent_units
    beneficial = ideal_speedup >= min_speedup
    per_call_gain = ambient_units - latent_units
    break_even = discovery_units / per_call_gain if per_call_gain > 0 else None
    return CostReport(
        ambient_units=ambient_units,
        latent_units=latent_units,
        discovery_units=discovery_units,
        ideal_speedup=ideal_speedup,
        beneficial=beneficial,
        break_even_calls=break_even,
    )


def _verdict(certificate: Certificate, cost: CostReport) -> tuple[bool, str]:
    if not certificate.holds:
        return False, "no certified latent under this basis and tolerance"
    if not cost.beneficial:
        return False, "certified representation exists but has no useful cost advantage"
    kind = "exact" if certificate.kind == "exact" else "bounded-error"
    return True, f"{kind} latent replacement with certified cost advantage"


# --------------------------------------------------------------------------
# Step 5: orchestration
# --------------------------------------------------------------------------


def _compile_boolean_sketched(
    F: Callable[[int], int] | np.ndarray,
    m_bits: int,
    tolerance: float,
    vectorized: bool,
    min_speedup: float,
    observables: list | None,
    n_samples: int,
    threshold: float,
    seed: int,
) -> CompileResult:
    """Compile a Boolean map by sketched (sampled) Walsh-Koopman discovery.

    Avoids building the 2^m x 2^m operator: heavy Walsh coefficients of each
    chosen observable are found by Kushilevitz-Mansour sampling, so the map can
    exceed the exact-enumeration bit limit. The certificate is probabilistic
    (energy captured from samples), not exact.
    """
    if m_bits > MAX_BOOLEAN_SKETCH_BITS:
        raise ValueError(
            f"m_bits={m_bits} exceeds sketched limit {MAX_BOOLEAN_SKETCH_BITS} "
            "(executor allocates a length-2^m vector)."
        )
    mask = (1 << m_bits) - 1
    if isinstance(F, np.ndarray):
        table = np.asarray(F, dtype=np.int64) & mask
        F_eval: Callable[[np.ndarray], np.ndarray] = lambda x: table[x]
        vectorized = True
    else:
        F_eval = F
    if observables is None:
        observables = [1 << i for i in range(m_bits)]  # singleton output-bit characters

    rng = np.random.default_rng(seed)
    rows, cols, vals, residuals, relative_error, n_evals = sketched_boolean_latent(
        F_eval, m_bits, observables, tolerance, n_samples, threshold, vectorized, rng
    )

    n = 1 << m_bits
    n_cols = len(observables)
    nnz = len(vals)
    certificate = _certificate(relative_error, tolerance, verified_on="sampled_walsh")
    cost = _cost_report(
        ambient_units=float(n_cols * n * max(1.0, float(np.log2(n)))),  # dense FWHT per column
        latent_units=float(max(1, nnz)),
        discovery_units=float(max(1, n_evals)),  # sampled evaluations, independent of 2^m
        min_speedup=min_speedup,
    )
    accepted, reason = _verdict(certificate, cost)

    mean_terms = nnz / n_cols if n_cols else 0.0
    representation = {
        "m_bits": m_bits,
        "nnz": nnz,
        "mean_terms_per_column": float(mean_terms),
        "coo_rows": np.asarray(rows, dtype=int),
        "coo_cols": np.asarray(cols, dtype=int),
        "coo_values": np.asarray(vals, dtype=float),
        "shape": (n, n),
        "method": "sketched",
        "observables": [int(s) for s in observables],
        "columns_compiled": n_cols,
        "per_column_residual": residuals,
        "n_samples": n_samples,
        "threshold": threshold,
        "discovery_evaluations": int(n_evals),
    }
    latent_complexity = float(max(0.0, (mean_terms - 1.0) / (n - 1.0))) if n > 1 else 0.0

    return CompileResult(
        accepted=accepted,
        basis="walsh_koopman",
        dimension=n,
        tolerance=tolerance,
        latent_complexity=latent_complexity,
        certificate=certificate,
        cost=cost,
        reason=reason,
        representation=representation,
    )


def compile_boolean(
    F: Callable[[int], int] | np.ndarray,
    m_bits: int,
    tolerance: float = 0.0,
    vectorized: bool = False,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    method: str = "exact",
    observables: list | None = None,
    n_samples: int = 2000,
    threshold: float = 1e-2,
    seed: int = 0,
) -> CompileResult:
    """Compile a finite Boolean map into a certified sparse Walsh-Koopman latent.

    ``method="exact"`` builds the full 2^m x 2^m operator (exact certificate, but
    bounded by ``MAX_BOOLEAN_BITS``). ``method="sketched"`` discovers heavy Walsh
    coefficients per observable by sampling (probabilistic certificate), allowing
    larger maps.
    """
    if method not in ("exact", "sketched"):
        raise ValueError(f"method must be 'exact' or 'sketched', got {method!r}")
    if method == "sketched":
        return _compile_boolean_sketched(
            F, m_bits, tolerance, vectorized, min_speedup, observables, n_samples, threshold, seed
        )

    if m_bits > MAX_BOOLEAN_BITS:
        raise ValueError(
            f"m_bits={m_bits} exceeds exact-enumeration limit {MAX_BOOLEAN_BITS}; "
            "use method='sketched' for larger maps."
        )
    if isinstance(F, np.ndarray):
        truth_table = np.asarray(F, dtype=np.int64) & ((1 << m_bits) - 1)
    else:
        truth_table = boolean_truth_table(F, m_bits, vectorized=vectorized)

    n = 1 << m_bits
    K = walsh_koopman_operator(truth_table, m_bits)
    K_sparse = sparse_column_latent(K, tolerance)

    relative_error = _relative_frobenius(K, K_sparse)
    certificate = _certificate(relative_error, tolerance, verified_on="full_state_space")

    nnz = int(np.count_nonzero(K_sparse))
    cost = _cost_report(
        ambient_units=float(n * n),
        latent_units=float(nnz),
        discovery_units=float(n * n),  # building the exact operator costs n^2
        min_speedup=min_speedup,
    )
    accepted, reason = _verdict(certificate, cost)

    rows, cols = np.nonzero(K_sparse)
    representation = {
        "m_bits": m_bits,
        "nnz": nnz,
        "mean_terms_per_column": float(nnz / n),
        "coo_rows": rows.astype(int),
        "coo_cols": cols.astype(int),
        "coo_values": K_sparse[rows, cols].astype(float),
        "shape": (n, n),
    }

    return CompileResult(
        accepted=accepted,
        basis="walsh_koopman",
        dimension=n,
        tolerance=tolerance,
        latent_complexity=normalized_latent_complexity(K),
        certificate=certificate,
        cost=cost,
        reason=reason,
        representation=representation,
    )


def compile_matrix(
    A: np.ndarray,
    tolerance: float = 0.0,
    value_width: int = 1,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    method: str = "exact",
    max_rank: int | None = None,
    oversample: int = 10,
    n_iter: int = 2,
    seed: int = 0,
) -> CompileResult:
    """Compile a linear operator into a certified low-rank spectral latent.

    ``method="exact"`` uses a full SVD (discovery ~ n^3). ``method="sketched"``
    uses randomized range finding (discovery ~ n^2 * sketch_size), keeping the
    certificate exact for the produced latent. Sketched discovery is the lever
    that scales the compiler to large operators where a full SVD is infeasible.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square 2D operator, got shape {A.shape}")
    if method not in ("exact", "sketched"):
        raise ValueError(f"method must be 'exact' or 'sketched', got {method!r}")
    n = A.shape[0]

    if method == "sketched":
        rng = np.random.default_rng(seed)
        U, s, Vt, r, relative_error, discovery_cols = sketched_low_rank_latent(
            A, tolerance, max_rank=max_rank, oversample=oversample, n_iter=n_iter, rng=rng
        )
        verified_on = "sketched_frobenius"
        discovery_units = float(n * n * max(discovery_cols, 1.0))
    else:
        U, s, Vt, r = low_rank_latent(A, tolerance)
        approx = (U * s) @ Vt
        relative_error = _relative_frobenius(A, approx)
        verified_on = "frobenius"
        discovery_units = float(n * n * n)  # SVD discovery ~ n^3

    certificate = _certificate(relative_error, tolerance, verified_on=verified_on)

    d = max(1, value_width)
    cost = _cost_report(
        ambient_units=float(n * n * d),
        latent_units=float(2 * n * r * d),
        discovery_units=discovery_units,
        min_speedup=min_speedup,
    )
    accepted, reason = _verdict(certificate, cost)

    # latent complexity for matrices: normalized effective rank.
    representation = {
        "rank": r,
        "U": U,
        "s": s,
        "Vt": Vt,
        "shape": (n, n),
        "value_width": d,
        "method": method,
    }

    return CompileResult(
        accepted=accepted,
        basis="spectral_svd",
        dimension=n,
        tolerance=tolerance,
        latent_complexity=float(r / n),
        certificate=certificate,
        cost=cost,
        reason=reason,
        representation=representation,
    )


def latent_compile(
    target,
    m_bits: int | None = None,
    tolerance: float = 0.0,
    vectorized: bool = False,
    value_width: int = 1,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    method: str = "exact",
    max_rank: int | None = None,
    oversample: int = 10,
    n_iter: int = 2,
    seed: int = 0,
    observables: list | None = None,
    n_samples: int = 2000,
    threshold: float = 1e-2,
) -> CompileResult:
    """Compile a computation into a certified, cost-beneficial latent representation.

    Dispatch:
      - 2D square ``np.ndarray``        -> low-rank spectral latent (matrix view).
      - callable ``F`` with ``m_bits``  -> sparse Walsh-Koopman latent (Boolean view).
      - 1D ``np.ndarray`` truth table   -> Boolean view (length must be 2^m_bits).

    ``method="sketched"`` discovers the latent by sampling instead of full
    enumeration: randomized range finding for matrices, Kushilevitz-Mansour
    heavy-hitter search for Boolean maps.
    """
    if isinstance(target, np.ndarray) and target.ndim == 2:
        return compile_matrix(
            target,
            tolerance=tolerance,
            value_width=value_width,
            min_speedup=min_speedup,
            method=method,
            max_rank=max_rank,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed,
        )

    if isinstance(target, np.ndarray) and target.ndim == 1:
        if m_bits is None:
            inferred = int(round(np.log2(target.size)))
            if (1 << inferred) != target.size:
                raise ValueError("truth-table length must be a power of two")
            m_bits = inferred
        return compile_boolean(
            target,
            m_bits,
            tolerance=tolerance,
            min_speedup=min_speedup,
            method=method,
            observables=observables,
            n_samples=n_samples,
            threshold=threshold,
            seed=seed,
        )

    if callable(target):
        if m_bits is None:
            raise ValueError("m_bits is required when compiling a callable Boolean map")
        return compile_boolean(
            target,
            m_bits,
            tolerance=tolerance,
            vectorized=vectorized,
            min_speedup=min_speedup,
            method=method,
            observables=observables,
            n_samples=n_samples,
            threshold=threshold,
            seed=seed,
        )

    raise TypeError(f"unsupported compile target of type {type(target)!r}")
