---
title: Latent Compiler
type: Guide
status: Active
created: 2026-06-05
last_updated: 2026-06-06
author: Dr. Tamás Nagy
agent: GPT-5.5
---

# Latent Compiler

A certifying compiler that answers one question about a computation:

> Is there a basis in which this computation is cheap, and can we certify it —
> or can we prove there is no cheap basis?

This is the reusable system form of the `meta_algorithmic_latent` research
program. The research experiment (`experiments/algorithmic_latent_bit_dynamics.py`)
is the pilot; this package is the engine.

## The contract

```
computation  ->  operator extraction  ->  latent discovery
             ->  correctness certificate  ->  cost certificate
             ->  accept only if certified AND cost-beneficial
             ->  lower to a runnable kernel
```

A latent is **accepted** only when both hold:

1. **Certificate** — the sparse/low-rank representation reproduces the original
   exactly (`kind="exact"`) or within tolerance (`kind="bounded"`). Boolean maps
   are verified on the full `{0,1}^m` state space; matrices via the relative
   Frobenius truncation bound (rank chosen so the certified error meets the
   tolerance by construction).
2. **Cost advantage** — the latent execution is at least `min_speedup` (default
   2x) cheaper than ambient execution.

An exact representation that is not cheaper is **rejected** — this is what
prevents random or fully diffused cryptographic maps from being advertised as
"compiled".

## Three views

| Input | Basis | Discovery | Certificate | Cost |
|---|---|---|---|---|
| Boolean map `F: {0,1}^m -> {0,1}^m` | Walsh-Koopman | per-column sparse (full enum or KM-sketched) | full state space (exact) / sampled (probabilistic) | `nnz` vs `n^2` |
| Linear operator `A` (n x n) | spectral SVD | rank-`r` truncation (full SVD or sketched) | relative Frobenius bound | `2nr` vs `n^2` |
| Signal / series `x_k` | Hankel `H[i,j]=x_{i+j}` | progressive SVD (PME/CASLA) | AAK tail `s_r/s_0` + Weyl analytic confidence-sequence band | `r` modes vs slow series |

### Discovery: exact vs sketched (matrix view)

`compile_matrix(..., method="exact")` factors the whole operator (full SVD,
`O(n^3)`). `method="sketched"` uses randomized range finding: it probes the
operator on a few random directions, orthonormalizes the response, and factors
the small projected operator (`O(n^2 * sketch_size)`). The sketch size doubles
until the achievable error meets the tolerance, or spans the space (recovering
the exact factorization).

The certificate stays **exact** for the produced latent: given the sketch basis
`Q`, `||A - A_r||_F^2 = (||A||_F^2 - ||Q^T A||_F^2) + sum_{i>=r} s_i^2`, computed
exactly without ever forming the full SVD. On a 1500x1500 smooth-kernel
operator, sketched discovery finds the identical rank-15 latent with the
identical certified error ~44x faster than the full SVD.

### Discovery: exact vs sketched (Boolean view)

`compile_boolean(..., method="exact")` builds the full `2^m x 2^m` Walsh-Koopman
operator (exact certificate, capped near `m=14`). `method="sketched"` removes the
cap by never forming the operator: for each observable `S` (default = single
output-bit characters), `g_S(x) = (-1)^<S, F(x)>` is a +/-1 function with total
Walsh energy 1, and its heavy coefficients are found by Kushilevitz-Mansour
search (split the frequency space one bit at a time, estimate bucket energy by
sampling, prune light buckets). Survivors are the sparse latent; captured energy
gives the residual.

The Boolean sketch certificate is **probabilistic** (not exact): over an
exponential state space no sampling can certify exactly, so the residual
`1 - sum_j c_j^2` is a sampled estimate. On an 18-bit GF(2)-linear map (dense
operator `2^36` entries, infeasible) it recovers exactly one term per output bit
with zero held-out error, touching ~2M sampled evaluations; a diffusive mixing
hash at the same width is correctly rejected.

### Signal / Hankel view: the latent of the latent

A slowly converging series (or a black-box trajectory) is itself a latent of a
dynamics — the motivating case is the Sundman series for the three-body problem,
whose coefficients are infeasible to compute. The signal view asks whether that
representation has its own latent: for `x_k = sum_j a_j z_j^k`, the Hankel matrix
`H[i,j] = x_{i+j}` has rank equal to the number of modes (Kronecker), so a
low-rank Hankel means the series **resums** to a fast finite-mode model. Two
facts make it rigorous and *progressive*:

- **AAK theory:** the optimal rank-`r` Hankel-norm approximation error is exactly
  `s_r` (the `(r+1)`-th singular value), so `s_r / s_0` is a principled relative
  certificate, not a heuristic.
- **Weyl + analytic confidence sequence:** the Hankel is estimated from data with
  per-sample noise `eta`. `analytic_noise_floor(...)` is a closed-form
  high-probability bound on `||noise Hankel||_2`,
  `C * eta * sqrt(m * (log m + log(K/delta)))`, combining the Meckes mean scaling
  `sqrt(m log m)` with a sub-Gaussian / Talagrand tail (`||E||_2` is
  `sqrt(min(p,q))`-Lipschitz in the i.i.d. entries). The `log(K/delta)` term is a
  union bound over `K` checkpoints, so the band is a *time-uniform* confidence
  sequence valid simultaneously across a whole progressive schedule. By Weyl the
  true singular values lie within the floor; relative to a signal whose
  `sigma_1 ~ sqrt(p*q)`, the band shrinks `~1/sqrt(N)`. The constant `C` is
  Monte-Carlo-calibrated to dominate the empirical `(1-delta)` quantile (regression
  tested), and the analytic path needs no per-call simulation — it is `O(1)` and
  deterministic, ~25x faster than the Monte-Carlo reference floor.
- **Decision calculus (formal PME link):** for each rank `r` the gate forms a
  certified interval `[L_r, U_r]` on the true relative tail and decides
  `tail <= tolerance` via `decide_interval(...)` — `provably_below` /
  `provably_above` / `undecided`. These are byte-for-byte the semantics of
  `spectral_fenton...decide_interval_threshold` (the PME/CASLA verified anytime
  decision calculus), verified equal in `test_decide_interval_matches_pme_calculus`.
  So the Hankel gate is a literal instance of PME's calculus: sound (never wrong
  while the interval is valid), monotone (a narrower band never un-decides), and
  stable (a decided verdict stays decided as data grows).

`hankel_decision(...)` returns `resummable` / `no_low_rank_latent` (chaos /
hardness) / `undecided`; `estimate_modes(...)` recovers the poles by matrix
pencil (ESPRIT) and `resummation_extrapolation_error(...)` fits on a prefix and
predicts the unseen tail. Demonstrated in
`experiments/progressive_hankel_latent_demo.py`: a finite-mode signal is certified
resummable at the true rank (held-out error `1e-3`); a natural-boundary signal is
certified hardness; the figure-eight kinetic energy `K(t)` resums while the
bounded-chaotic Pythagorean `K(t)` is certified `no_low_rank_latent`.

## Application: certified Gaussian-process inference

`experiments/latent_gp_certified_demo.py` is the first widely-painful
application. GP regression (Bayesian optimization, AutoML, geostatistics) needs
`(K + sigma^2 I)^{-1} y` and `log det(K + sigma^2 I)` — `O(n^3)`. People scale it
with Nystrom / inducing points, but those approximations carry **no error
guarantee**. The latent compiler discovers a certified rank-`r` factor
`K_r = W W^T`, then runs exact low-rank GP algebra (Woodbury solve + matrix
determinant lemma), `O(n r^2)`.

The key insight is the **noise-floor rule**: certifying the *operator* is not
enough. The resolvent `(K+sigma^2 I)^{-1}` amplifies exactly the discarded
subspace, so a fixed 1% kernel error with a small `sigma^2` produces a wildly
wrong posterior. Tying the truncation to the noise floor — choose `r` so
`||K - K_r||_F <= eps * sigma^2` — makes the GP *prediction* certified, via the
resolvent-perturbation bound

```
||alpha - alpha_hat||_2 / ||alpha||_2  <=  ||K - K_r||_2 / sigma^2  <=  ||K - K_r||_F / sigma^2  =  eps.
```

Measured (n=4000, smooth RBF, sigma^2=0.05, eps=0.05): compiled to rank 239,
**51x per-solve speedup** in the fixed-kernel many-RHS regime, posterior mean
relative error 2.1e-5 (certified bound 5e-2 holds), marginal-LL recovered;
Nystrom at the same rank is ~1000x less accurate and carries no certificate. A
rough (near-full-rank) kernel is **rejected** (rank = n), so we never ship a bad
low-rank GP. (Discovery via sketched SVD ~ one Cholesky, so the win is in the
repeated-solve regime — fixed kernel, many right-hand sides / online targets.)

## The noise-floor rule, generalized (`resolvent.py`)

The GP insight is not GP-specific. Many widely used computations end in a
*regularized resolvent solve*

```
x = (A + delta * I)^{-1} b,    A symmetric PSD,   delta > 0,
```

where `delta` is a known positive floor: the GP noise variance, the ridge
parameter in kernel ridge regression, the Tikhonov parameter in a linear inverse
problem, the damping term in a Levenberg-Marquardt / damped-Newton step, the
teleport mass in a regularized graph diffusion. For **all** of them the same
one-line theorem holds (`E = A - A_r`, both PSD-shifted, so `A_hat >= delta I`):

```
x_hat - x = A_hat^{-1} E x   =>   ||x - x_hat|| / ||x||  <=  ||E||_2 / delta  <=  ||E||_F / delta.
```

So the **noise-floor rule** is: pick the rank so `||A - A_r||_F <= eps * delta`,
and the relative output error of the solve is certified `<= eps`, no matter how
ill-conditioned `A` is. The floor `delta` is the amplification budget the rule
spends deliberately. `certify_resolvent(A, floor, eps)` applies the rule and
returns the PSD factor `W` (`A_r = W W^T`) plus the certified output bound;
`resolvent_solve(W, b, floor)` does the `O(n r^2)` Woodbury solve.

Measured (`experiments/noise_floor_certified_demo.py`): kernel ridge (n=3000)
compiles to rank 199, 56x per-solve speedup, certified bound 5e-2 holds (actual
9e-4); a Tikhonov inverse (n=1800) compiles to rank 35, 130x; a **floor sweep**
on one kernel shows the selected rank growing monotonically as the floor shrinks
(122 -> 157 -> 194 -> 231 for delta = 1.0 -> 0.3 -> 0.1 -> 0.03) with the bound
holding at every floor; a rough kernel with a tiny floor is **rejected**. The
scope is honest: it requires `A` symmetric PSD and an explicit positive floor —
an unregularized solve has no finite amplification budget and is out of scope.

## Query-only discovery (`operator.py`)

The matrix view needs the dense operator (the exact certificate uses `||A||_F`
and the full SVD). But many operators exist only as a *matvec* `v -> A v` — a PDE
solve, a graph propagation, an attention map, a simulator linearization. For
those, both discovery and certification run from matvec queries alone:

* **Discovery** — randomized subspace iteration (Halko-Martinsson-Tropp): only
  `matvec` (and `rmatvec` for non-symmetric `A`) are ever called, so `A` is
  never formed.
* **Certification** — the residual operator `R = (I - QQ^T)A` is itself
  matvec-accessible (`R w = A w - Q(Q^T(A w))`), so we probe it with fresh
  Gaussians and bound the spectral residual:

```
P[ ||(I - QQ^T) A||_2 <= alpha * max_i ||R w_i||_2 ] >= 1 - 10^{-p},   alpha = 10*sqrt(2/pi).
```

This is a **probabilistic spectral** certificate (failure prob `10^{-p}`),
analogous to the Boolean sketch — without the full operator no procedure can
certify exactly. Cost is counted in **matvecs**, the real currency here.

Measured (`experiments/query_only_latent_demo.py`): a planted rank-20 operator
behind a matvec is recovered at rank 20 in 480+6 matvecs, certified residual
6e-14, held-out spectral error 6e-16; a smooth kernel with a 50x-expensive matvec
compiles to rank 5 (6000x per-apply saving, break-even ~646 calls), certified
relative spectral residual 1.73e-2 holding against the true 1.73e-2; a full-rank
random operator is **rejected**. Note the certificate is on the *spectral* norm:
it bounds the error on the dominant subspace, not the per-output error for inputs
living in the discarded low-energy subspace.

## Modules

| File | Role |
|---|---|
| `core.py` | operator extraction, discovery, certification, cost, `latent_compile` orchestrator, result types |
| `hankel.py` | signal/Hankel view: progressive certified-anytime rank decision, AAK certificate, matrix-pencil mode estimation, resummation |
| `resolvent.py` | noise-floor rule: certified low-rank substitution for regularized resolvent solves `(A + delta I)^{-1} b` (GP, kernel ridge, Tikhonov, damped Newton, graph diffusion) |
| `operator.py` | query-only (matrix-free) view: discover + certify a low-rank latent from matvec access alone (randomized subspace iteration + HMT a-posteriori spectral bound) |
| `lowering.py` | runnable NumPy executor, emitted Python source, round-trip verification |
| `native.py` | machine-code lowering: emit C, compile with `clang -O3 -march=native`, load via `ctypes` |
| `cli.py` / `__main__.py` | command-line interface |
| `__init__.py` | public API |

## API

```python
import numpy as np
import latent_compiler as lc

# Matrix view
A = np.random.default_rng(0).standard_normal((128, 3)) @ \
    np.random.default_rng(1).standard_normal((3, 128))
res = lc.latent_compile(A, tolerance=1e-8, value_width=16)
res.accepted, res.representation["rank"], res.cost.ideal_speedup

# Sketched discovery for large operators (exact certificate, cheap discovery)
res = lc.latent_compile(A, tolerance=1e-3, value_width=16,
                        method="sketched", max_rank=16)

# Boolean view (truth table or callable + m_bits)
rot = lambda x, m=10: ((x << 1) | (x >> (m - 1))) & ((1 << m) - 1)
res = lc.latent_compile(rot, m_bits=10, vectorized=True)
res.summary()

# Sketched Boolean discovery beyond the exact bit limit (probabilistic certificate)
res = lc.latent_compile(rot, m_bits=18, method="sketched",
                        tolerance=0.05, threshold=0.25, vectorized=True)

# Signal / Hankel view: the latent of the latent (certified-anytime)
k = np.arange(500)
x = np.cos(2 * np.pi * 0.02 * k) + 0.5 * np.cos(2 * np.pi * 0.05 * k)
v = lc.hankel_decision(x, tolerance=5e-2, eta=1e-3, max_rank=20)
v.verdict, v.certified_rank                       # ("resummable", 4)
lc.resummation_extrapolation_error(x, rank=4)     # held-out prediction error

# Noise-floor rule: certified low-rank substitution for a regularized solve
# (K = any symmetric PSD operator, e.g. a kernel/Gram matrix; b a target vector)
cert = lc.certify_resolvent(K, floor=0.1, eps=0.05, method="exact")
if cert.accepted:                                 # rank chosen at the noise floor
    x = lc.resolvent_solve(cert.factor, b, floor=0.1)
    cert.rel_output_bound                         # certified <= eps

# Lower an accepted latent to a runnable kernel
kernel = lc.build_executor(res)
source = lc.emit_source(res)

# Lower to native machine code (clang -O3 -march=native), load via ctypes
if lc.compiler_available():
    native = lc.compile_native(res)      # emits C, compiles, returns a callable
    y = native(np.random.standard_normal(res.representation["shape"][0]))
```

## Native lowering — what it buys (measured, honest)

`native.py` closes the "compile to machine code" loop: `emit_c_source(res)`
generates a self-contained C kernel for the fixed discovered structure
(sparse scatter for Boolean, fused `y = U(s*(Vt x))` for matrix), and
`compile_native(res)` compiles it with `clang -O3 -march=native` and loads it
through `ctypes`. Measured on
`experiments/latent_native_compile_demo.py`:

| Kernel | latent vs dense | native vs NumPy executor |
|---|---|---|
| RBF matrix (n=1024, rank 32, single vector) | ~6.4x | ~1.1x |
| rotate-left GF(2) bit map (n=4096) | ~445x | ~0.9x |

The large speedups are the **algorithmic** latent doing its job against the
dense baseline. Against the NumPy executor, native *ties* at these sizes
because NumPy's primitives (BLAS gemv, the C-level `np.add.at` scatter) are
already compiled and the `ctypes` boundary costs ~a microsecond. Native
lowering's value is **deployability** — a standalone machine-code artifact with
no Python/NumPy runtime — and the path to SIMD/GPU/fixed-size kernels, not a
NumPy speed race. It is not a claim to beat tuned BLAS on large dense GEMM.

## CLI

```bash
PYTHONPATH=src python3 -m latent_compiler boolean --map all --m-bits 10 --tolerance 0
PYTHONPATH=src python3 -m latent_compiler matrix --input op.npz --key A --tolerance 1e-6 --value-width 16
```

## Lifecycle

- **CREATE** — `PYTHONPATH=src python3 -m py_compile src/latent_compiler/*.py`
- **INTEGRATE** — registered in `_brain/registries/tool_registry.yaml` (analytics)
  and `_brain/MAP.md`; discoverable as `import latent_compiler`.
- **SELF-IMPROVE** — health signal: `tests/test_latent_compiler.py` (41 tests)
  passes, including the analytic-floor domination guarantee, PME decision-calculus
  equivalence, and native round-trip correctness (matrix + Boolean vs dense, skipped
  if no C compiler). Next: tighten the Boolean probabilistic certificate; sharpen the
  analytic Hankel constant `C` toward the empirical tail (currently a conservative
  1.3); extend native lowering to SIMD/batched/fixed-size kernels (and GPU) where it
  can beat — not just match — the NumPy executor.

## Scope

This is an exact, small-state / dense-matrix system. It fails loudly above
`MAX_BOOLEAN_BITS = 14` rather than silently sampling. Large-state sampled
discovery with probabilistic or SMT certificates is the next phase.
