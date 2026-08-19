"""
Certified low-rank Gaussian-process inference via the latent compiler.

The pain: GP regression (Bayesian optimization, AutoML, geostatistics) needs
``(K + sigma^2 I)^{-1} y`` and ``log det(K + sigma^2 I)`` for an n x n kernel
matrix K — O(n^3) per hyperparameter evaluation. Practitioners scale it with
Nystrom / inducing points / random features, but those approximations come with
**no error guarantee**: you cannot tell whether the posterior you shipped is
trustworthy.

What the latent compiler adds: for a smooth kernel, K is genuinely low rank
(fast-decaying spectrum). ``latent_compile`` discovers a rank-r factor
K_r = W W^T with a **certified** relative-Frobenius error e = ||K - K_r||_F,
and *refuses* if the rank gives no cost advantage. We then run exact low-rank GP
algebra:

  * solve via Woodbury:   (sigma^2 I + W W^T)^{-1}
  * log-det via the matrix determinant lemma.

Both are O(n r^2 + r^3). Crucially, the certified operator error propagates to a
**certified bound on the GP posterior** via a resolvent perturbation argument:

  A = K + sigma^2 I,  A_hat = K_r + sigma^2 I,  E = K - K_r,
  alpha = A^{-1} y,    alpha_hat = A_hat^{-1} y
  ||alpha - alpha_hat||_2 <= ||A_hat^{-1}|| * ||E||_2 * ||A^{-1}|| * ||y||_2
                          <= e * ||y||_2 / sigma^4        (since ||A^{-1}|| <= 1/sigma^2,
                                                            ||E||_2 <= ||E||_F = e)

The sharp *relative* form (using alpha - alpha_hat = A_hat^{-1} E alpha):

  ||alpha - alpha_hat||_2 / ||alpha||_2 <= ||A_hat^{-1}||_2 ||E||_2
                                        <= ||E||_2 / sigma^2 <= ||E||_F / sigma^2.

This is the key design rule: the truncation must sit *below the noise floor*.
We pick the rank so ||K - K_r||_F <= eps * sigma^2, which makes the certified
relative posterior error <= eps. Truncating arbitrarily (e.g. a fixed 1% kernel
error with a tiny sigma^2) is *unsafe* — the resolvent amplifies exactly the
discarded subspace. Tying tolerance to sigma^2 is what makes the GP prediction
certified, not just the kernel.

The demo compares, on the SAME kernel:
  1. exact GP (Cholesky),
  2. certified low-rank GP (this method),
  3. Nystrom GP at the same rank (uncertified baseline),
and adds a short-lengthscale near-full-rank control that the compiler rejects
(so we never ship a bad low-rank GP).

Output: forge/meta_algorithmic_latent/latent_gp_certified_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import latent_compiler as lc  # noqa: E402

OUTPUT = ROOT / "forge" / "meta_algorithmic_latent" / "latent_gp_certified_results.json"


def rbf(X: np.ndarray, Z: np.ndarray, lengthscale: float) -> np.ndarray:
    sx = np.sum(X**2, axis=1)
    sz = np.sum(Z**2, axis=1)
    d2 = sx[:, None] + sz[None, :] - 2.0 * (X @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-d2 / (2.0 * lengthscale**2))


def _time_best(fn, repeats: int = 3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def exact_gp(K: np.ndarray, y: np.ndarray, sigma2: float):
    n = K.shape[0]
    A = K + sigma2 * np.eye(n)

    def solve():
        c, low = cho_factor(A, lower=True)
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(c))))
        return alpha, logdet

    t, (alpha, logdet) = _time_best(solve)
    mll = -0.5 * float(y @ alpha) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return {"time_s": t, "alpha": alpha, "logdet": logdet, "mll": mll}


def lowrank_gp_from_factor(W: np.ndarray, y: np.ndarray, sigma2: float):
    """Exact GP algebra for K_r = W W^T via Woodbury + determinant lemma."""
    n, r = W.shape

    def solve():
        WtW = W.T @ W                       # r x r
        M = np.eye(r) + WtW / sigma2        # I_r + sigma^-2 W^T W
        cM, lowM = cho_factor(M, lower=True)
        Wty = W.T @ y                       # r
        # alpha = sigma^-2 y - sigma^-4 W M^{-1} (W^T y)
        alpha = y / sigma2 - (W @ cho_solve((cM, lowM), Wty)) / (sigma2 * sigma2)
        logdet = 2.0 * n * np.log(np.sqrt(sigma2)) + 2.0 * np.sum(np.log(np.diag(cM)))
        return alpha, logdet

    t, (alpha, logdet) = _time_best(solve)
    mll = -0.5 * float(y @ alpha) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return {"time_s": t, "alpha": alpha, "logdet": logdet, "mll": mll}


def solver_factor_relative_frobenius_bound(
    Kfro: float, U: np.ndarray, s: np.ndarray, Vt: np.ndarray, discovery_error: float
) -> tuple[float, float]:
    r"""Certify the actual PSD solver factor without an additional dense multiply.

    The compiler certifies $\|K-U\operatorname{diag}(s)V^T\|_F/\|K\|_F$.
    Woodbury uses $U\operatorname{diag}(s)U^T$, so the triangle inequality
    gives a valid bound after adding the normalized factorization gap.  This is
    intentionally conservative, but avoids an $O(n^2r)$ verification step that
    would defeat the repeated-solve purpose of the demonstration.
    """
    # U has orthonormal columns, so
    # ||U diag(s) (Vt - U.T)||_F = ||diag(s) (Vt - U.T)||_F.
    # This avoids materializing an n-by-n matrix merely to certify the factor.
    gap_rel = float(np.linalg.norm(s[:, None] * (Vt - U.T), "fro") / Kfro)
    return float(discovery_error + gap_rel), gap_rel


def certified_lowrank_gp(K, y, sigma2, eps, method="sketched", max_rank=512):
    """Discover a rank-r latent at the noise floor (||K-K_r||_F <= eps*sigma^2)."""
    n = K.shape[0]
    Kfro = float(np.linalg.norm(K, "fro"))
    # Tie the relative-Frobenius tolerance to the noise floor sigma^2.
    tol_rel = eps * sigma2 / Kfro

    t0 = time.perf_counter()
    if method == "sketched":
        res = lc.latent_compile(
            K, tolerance=tol_rel, value_width=1, method="sketched", max_rank=max_rank, seed=0
        )
    else:
        res = lc.latent_compile(K, tolerance=tol_rel, value_width=1)
    discovery_time = time.perf_counter() - t0

    rec: dict = {
        "accepted": res.accepted,
        "rank": res.representation.get("rank"),
        "eps_target": eps,
        "tol_rel_at_noise_floor": tol_rel,
        "discovery_rel_fro_error": res.certificate.relative_error,
        "ideal_speedup": res.cost.ideal_speedup,
        "discovery_time_s": discovery_time,
        "reason": res.reason,
    }
    if not res.accepted:
        return rec, None

    # Symmetric PSD kernel: SVD factors give K_r = U diag(s) U^T; W = U sqrt(s).
    U = np.asarray(res.representation["U"])
    s = np.asarray(res.representation["s"])
    W = U * np.sqrt(np.maximum(s, 0.0))[None, :]

    # This is the certificate that belongs to the matrix actually passed to
    # Woodbury, K_r = W W^T = U diag(s) U^T.
    solver_rel_bound, factor_gap_rel = solver_factor_relative_frobenius_bound(
        Kfro, U, s, np.asarray(res.representation["Vt"]), res.certificate.relative_error
    )
    rec["certified_rel_fro_error"] = solver_rel_bound
    rec["solver_factor_gap_rel"] = factor_gap_rel
    rec["certificate_basis"] = "solver_factor_frobenius_triangle_bound"
    rec["accepted"] = bool(solver_rel_bound <= max(tol_rel, 1e-10))
    if not rec["accepted"]:
        rec["reason"] = "solver PSD factor fails the noise-floor residual gate"
        return rec, None

    gp = lowrank_gp_from_factor(W, y, sigma2)
    rec.update({"time_s": gp["time_s"], "logdet": gp["logdet"], "mll": gp["mll"]})

    # Certified relative posterior bound: ||alpha-alpha_hat||/||alpha|| <= ||E||_F/sigma^2.
    e = solver_rel_bound * Kfro
    rec["operator_error_fro"] = e
    rec["alpha_rel_certified_bound"] = e / sigma2
    return rec, gp["alpha"]


def nystrom_gp(K: np.ndarray, y: np.ndarray, sigma2: float, rank: int, seed: int = 0):
    """Uncertified Nystrom baseline at the same rank (random landmarks)."""
    n = K.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=rank, replace=False)
    Knm = K[:, idx]                      # n x m
    Kmm = K[np.ix_(idx, idx)]            # m x m
    # K_nys = Knm Kmm^{-1} Kmn = W W^T with W = Knm Kmm^{-1/2}.
    w, V = np.linalg.eigh(Kmm)
    w = np.maximum(w, 1e-12)
    Whalf = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    W = Knm @ Whalf
    gp = lowrank_gp_from_factor(W, y, sigma2)
    return {"time_s": gp["time_s"], "logdet": gp["logdet"], "mll": gp["mll"], "alpha": gp["alpha"]}


def run_case(label, n, d, lengthscale, sigma2, eps, method="sketched", max_rank=512, n_test=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Xt = rng.standard_normal((n_test, d))
    # smooth latent function + observation noise
    w = rng.standard_normal(d)
    f = np.sin(X @ w) + 0.5 * np.cos(2.0 * X[:, 0])
    y = f + np.sqrt(sigma2) * rng.standard_normal(n)

    K = rbf(X, X, lengthscale)
    Kt = rbf(Xt, X, lengthscale)         # n_test x n

    exact = exact_gp(K, y, sigma2)
    cert, alpha_hat = certified_lowrank_gp(K, y, sigma2, eps, method=method, max_rank=max_rank)

    rec = {
        "label": label,
        "n": n,
        "d": d,
        "lengthscale": lengthscale,
        "sigma2": sigma2,
        "eps": eps,
        "exact_time_s": exact["time_s"],
        "exact_mll": exact["mll"],
        "certified": {k: v for k, v in cert.items() if not k.startswith("_")},
    }

    if not cert["accepted"]:
        rec["verdict"] = "rejected_fall_back_to_exact"
        return rec

    # Predictions vs exact.
    m_exact = Kt @ exact["alpha"]
    m_cert = Kt @ alpha_hat
    post_rel_err = float(np.linalg.norm(m_exact - m_cert) / np.linalg.norm(m_exact))
    alpha_rel_err = float(np.linalg.norm(exact["alpha"] - alpha_hat) / np.linalg.norm(exact["alpha"]))

    # Nystrom baseline at the same rank.
    nys = nystrom_gp(K, y, sigma2, rank=cert["rank"], seed=seed + 1)
    nys_post_rel_err = float(np.linalg.norm(m_exact - (Kt @ nys["alpha"])) / np.linalg.norm(m_exact))

    speedup = exact["time_s"] / cert["time_s"] if cert["time_s"] > 0 else None
    speedup_with_discovery = exact["time_s"] / (cert["time_s"] + cert["discovery_time_s"])

    rec.update(
        {
            "verdict": "compiled",
            "speedup_solve_only": speedup,
            "speedup_incl_discovery_one_shot": speedup_with_discovery,
            "posterior_mean_rel_error": post_rel_err,
            "alpha_rel_actual_error": alpha_rel_err,
            "alpha_rel_certified_bound": cert["alpha_rel_certified_bound"],
            "alpha_bound_holds": bool(alpha_rel_err <= cert["alpha_rel_certified_bound"] + 1e-9),
            "mll_abs_error": abs(exact["mll"] - cert["mll"]),
            "nystrom_posterior_mean_rel_error": nys_post_rel_err,
            "nystrom_has_certificate": False,
        }
    )
    return rec


def main() -> None:
    cases = [
        run_case("smooth_rbf_lowrank", n=4000, d=4, lengthscale=2.5, sigma2=5e-2, eps=0.05,
                 method="sketched", max_rank=512, seed=0),
        run_case("rough_rbf_control", n=4000, d=4, lengthscale=0.25, sigma2=1e-2, eps=0.05,
                 method="exact", seed=3),
    ]

    payload = {
        "problem": "Gaussian-process inference: certified low-rank (K+sigma^2 I)^-1 and log-det",
        "method": "latent_compile -> Woodbury solve + matrix determinant lemma, with a propagated resolvent-perturbation certificate on the posterior",
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for c in cases:
        print(f"[{c['label']}]  n={c['n']} d={c['d']} lengthscale={c['lengthscale']} sigma2={c['sigma2']:g}")
        cert = c["certified"]
        if c["verdict"] != "compiled":
            print(f"    REJECTED (rank={cert['rank']}, ideal_speedup={cert['ideal_speedup']:.2f}x) "
                  f"-> fall back to exact GP. {cert['reason']}\n")
            continue
        print(f"    COMPILED rank={cert['rank']}  (rank chosen at noise floor: ||K-K_r||_F <= eps*sigma2, eps={c['eps']})")
        print(f"    certified ||K-K_r||_F/||K||_F={cert['certified_rel_fro_error']:.2e}")
        print(f"    exact GP {c['exact_time_s']*1e3:.0f}ms  ->  certified low-rank solve {cert['time_s']*1e3:.2f}ms"
              f"   speedup {c['speedup_solve_only']:.0f}x per solve (amortized: fixed kernel, many RHS / online targets)")
        print(f"    one-shot incl. discovery: {c['speedup_incl_discovery_one_shot']:.1f}x "
              f"(discovery ~= one Cholesky, so the win is in the repeated-solve regime)")
        print(f"    posterior mean rel error vs exact: {c['posterior_mean_rel_error']:.2e}   "
              f"marginal-LL abs error: {c['mll_abs_error']:.2e}")
        print(f"    CERTIFIED relative alpha-error bound {c['alpha_rel_certified_bound']:.2e}  "
              f">= actual {c['alpha_rel_actual_error']:.2e}  (holds={c['alpha_bound_holds']})")
        print(f"    Nystrom (same rank, NO certificate) posterior rel error: "
              f"{c['nystrom_posterior_mean_rel_error']:.2e}\n")

    try:
        display = OUTPUT.relative_to(ROOT)
    except ValueError:
        display = OUTPUT
    print(f"Wrote {display}")


if __name__ == "__main__":
    main()
