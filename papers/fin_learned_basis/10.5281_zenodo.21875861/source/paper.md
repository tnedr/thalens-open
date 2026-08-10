---
title: "Three Modes for Simulated Risk: A Data-Driven Spectral Basis for Portfolio Loss Distributions"
author: Tamás Nagy, Ph.D.
email: tnagyphd@gmail.com
orcid: 0009-0004-8079-4679
affiliation: Independent Researcher
website: https://thalens.org
type: Research Paper
style: quant_finance
status: Working Paper
version: "0.5"
date: "August 2026"
created: 2026-03-03
last_updated: "2026-08-10"
arxiv_category: q-fin.RM
msc2020: "62H25, 91G70"
keywords: [spectral risk, principal component analysis, portfolio loss distribution, reduced-order model, value at risk, simulation study]
build_ready: false
proof_status: "No publication-grade formal verification claim; legacy proof metadata requires source-bound reconciliation"
target_journal: Quantitative Finance / Journal of Financial Econometrics
depends_on:
  - fin_fenton_solved
  - fin_exact_var_computational
academic_potential: 52
business_potential: 68
novelty: "Applying PCA to Fourier-cosine coefficients of portfolio loss distributions (rather than to returns or yields) and demonstrating that the resulting spectral coefficient space is almost one-dimensional, yielding a 43x compression with sub-basis-point VaR error."
value_acceleration: "Run the full pipeline on 5+ years of real equity or credit portfolio data and show the 3-mode structure holds through at least one crisis period (e.g., COVID-2020 or a rate shock). A single empirical validation table would move this from 'interesting simulation study' to 'publishable result'."
content_hash: "18d419bf"
---

# Three Modes for Simulated Risk: A Data-Driven Spectral Basis for Portfolio Loss Distributions

Tamás Nagy, Ph.D.

ORCID: 0009-0004-8079-4679  
Independent Researcher  
tnagyphd@gmail.com  
https://thalens.org

August 2026

## Abstract

The Spectral Fenton Distribution represents a portfolio loss distribution with 128 Fourier coefficients. Under the deterministic simulated protocol reported here (5 assets, 250 trading days, seed 42), a single principal component captures 97.81\% of observed coordinate variation and three components capture 99.9996\%. Three coordinates --- $z_1(t)$, $z_2(t)$, $z_3(t)$ --- provide a 43$\times$ per-day coefficient compression with a maximum relative VaR reconstruction error of 0.0416\%. A basis fitted on days 1--200 has a maximum relative VaR error of 0.0189\% on days 201--250. The dominant mode $z_1$ is strongly associated with simulated VaR; economic interpretations of the remaining modes are tentative. The coordinates provide a descriptive reduced-order representation, not a calibrated detector or an empirical claim about real portfolios.

---

**Key Messages**

- 128 spectral coefficients compress to 3 numbers with $<$0.05\% VaR error (simulated data, $n = 5$ assets)
- Mode 1 alone captures 97.8\% of observed simulated coefficient variation; three modes capture 99.9996\%
- A 200-day fitted basis gives at most 0.0189\% relative VaR error on the 50-day simulated holdout
- The 3-coordinate dashboard visualizes simulated stress episodes via Mahalanobis distance; detection performance remains untested
- The learned basis offers a candidate reduced-order representation for Dynamic URRT, Bayesian Risk, and optimal transport

---

## 1. Introduction

### 1.1 The Dimensionality Question

The Spectral Fenton Distribution (Nagy, 2026, `fin_fenton_solved`) encodes a portfolio's loss distribution as $N = 128$ Fourier-cosine coefficients $A_0, \ldots, A_{127}$. This is already a dramatic compression from the infinite-dimensional space of probability distributions. But for real-time risk management, even 128 numbers may be too many: a risk manager needs to understand, at a glance, what changed today.

The question is: **how many independent ways can the distribution actually change?**

If the answer is 3, then the risk dashboard is three numbers. If the answer is 10, it is ten. The answer determines the intrinsic dimensionality of the risk monitoring problem.

### 1.2 The PCA Approach

Let $A(t) \in \mathbb{R}^{128}$ denote the coefficient vector on day $t$. Over $T$ trading days, we observe a $T \times 128$ matrix. Principal Component Analysis (PCA) extracts the dominant modes of variation:

$$
A(t) \approx \bar{A} + z_1(t) \cdot v_1 + z_2(t) \cdot v_2 + \cdots + z_r(t) \cdot v_r
$$

where $\bar{A}$ is the time-averaged coefficient vector, $v_1, \ldots, v_r$ are orthonormal directions in $\mathbb{R}^{128}$ (the principal modes), and $z_1(t), \ldots, z_r(t)$ are the coordinates on day $t$. The question reduces to: how large must $r$ be?

### 1.3 Related Work

PCA is a foundational tool in quantitative finance. The seminal application to interest-rate term structures by Litterman and Scheinkman (1991) showed that three factors --- level, slope, and curvature --- capture most yield-curve variation. Dai and Singleton (2000) connected low-dimensional factor structure to affine term-structure models. The present study changes the object being decomposed: each observation is a coefficient vector encoding a simulated portfolio loss distribution rather than a vector of yields or asset returns.

Our work differs in *what* is being decomposed: not asset returns, not yield curves, but the Fourier-cosine coefficients that encode a portfolio loss distribution. This is a finite-dimensional SVD on the coefficient matrix. Because the integration domain may vary with the daily cumulant rule, the analysis is an empirical comparison of coefficient coordinates rather than a fixed-function-space PCA; the daily domain is additional state. The key empirical finding is that 3 modes provide near-perfect reconstruction of the reported simulated coefficient trajectories, which parallels the yield curve result but concerns distributional coordinates rather than price dynamics.

### 1.4 Contribution

We find empirically that $r = 3$ captures 99.9996\% of observed simulated coefficient variation across 250 trading days:

1. **Compression**: 128 $\to$ 3 numbers (43$\times$) with mean VaR error 0.005\% (Section 3).
2. **Interpretation**: Mode 1 is strongly associated with VaR ($r = 0.998$); the interpretations of Modes 2--3 are tentative (Section 3.3).
3. **Dashboard**: each day is a point $(z_1, z_2, z_3) \in \mathbb{R}^3$. Mahalanobis distance visualizes simulated outliers, while detection performance remains open (Section 3.4).
4. **Holdout diagnostic**: the three-mode representation remains accurate on the declared chronological simulation holdout (Section 3.6); cross-portfolio and crisis robustness are outside this version's evidence.
5. **Unification**: the learned basis offers candidate reduced-order coordinates for Bayesian filtering (#7), temporal compression (#2), Schrödinger bridges (#11), and instanton paths (#13), subject to model-specific approximation validation.

This study is a companion to *Deterministic Log-CF Evaluation for Correlated Lognormal Sums*, *Exact Portfolio VaR Without Monte Carlo: The Eigen-COS Method*, *Noise-Free Risk: Deterministic VaR, ES, and Spectral Risk Measures*, and *The Anomaly Functional: Real-Time Arbitrage Detection via Spectral Risk Coefficients* (Nagy, 2026). These related working and Zenodo papers provide the distribution, deterministic-risk, and anomaly contexts; they do not certify the numerical results reported here.

---

## 2. Method

### 2.1 Data Generation

We simulate $T = 250$ trading days of an $n = 5$-asset portfolio with random seed 42. The implemented update is a discrete stochastic recurrence, not a calibrated continuous-time SDE. Initial volatilities are deterministic, $sigma_i(0)=0.15+0.1i/5$ for $i=0,\ldots,4$, and the equal portfolio weights remain fixed.

At each day, the code applies

$$
\sigma_i \leftarrow \operatorname{clip}\!\left(\sigma_i\exp(0.01Z_i),0.01,2\right),
\qquad
R \leftarrow 0.99R+0.01R_0+0.005E,
$$

where $Z_i$ are seeded standard-normal draws, $R_0$ has off-diagonal entries $0.3$, and $E$ is a symmetrized seeded Gaussian perturbation with zero diagonal. The implementation clips correlations to $[-0.99,0.99]$ and applies a positive-definiteness correction when necessary. In words:

- **Volatilities**: geometric Brownian motion with 1\% daily noise
- **Correlations**: a discrete 1% pull toward the 0.3 base-correlation matrix with a 0.5% symmetric daily perturbation, then a positive-definiteness correction if required
- **Weights**: constant (equal-weight portfolio, $w_i = 1/5$)

For each day $t$, the Eigen-COS algorithm (Nagy, 2026, `fin_exact_var_computational`) computes $A(t) \in \mathbb{R}^{128}$ using $N = 128$ Fourier-cosine terms and an integration domain $[a, b]$ determined by the cumulant-based rule of Fang and Oosterlee (2008). If this domain varies by day, the coefficient index refers to a day-specific basis; accordingly, the reported PCA is a coordinate-level empirical compression conditional on the domain rule, not a fixed-function-space analysis.

### 2.2 SVD Decomposition

Center the coefficient matrix: $\tilde{A}(t) = A(t) - \bar{A}$. Compute the Singular Value Decomposition:

$$
\tilde{A} = U \Sigma V^\top
$$

where $U$ is $T \times T$, $\Sigma$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots$, and $V$ is $128 \times 128$. The columns of $V$ are the principal modes $v_j$, and the coordinates are $z_j(t) = \tilde{A}(t)^\top v_j$.

### 2.3 Projection and Reconstruction

**Projection**: given a new coefficient vector $A^*$, compute

$$z_j = (A^* - \bar{A})^\top v_j, \quad j = 1, \ldots, r.$$

**Reconstruction**: from $r$ coordinates, recover

$$\hat{A} = \bar{A} + \sum_{j=1}^{r} z_j \cdot v_j.$$

The reconstruction error is $\lVert A^* - \hat{A} \rVert_2$.

---

## 3. Results

### 3.1 Variance Explained

Table 1 shows the cumulative variance explained by the first $r$ modes.

| Modes ($r$) | Variance explained | Incremental |
|---|---|---|
| 1 | 97.81\% | 97.81\% |
| 2 | 99.98\% | 2.17\% |
| 3 | 99.9996\% | 0.02\% |
| 5 | 100.00\% | $< 10^{-4}$\% |

A single mode captures 97.8\% of observed simulated coefficient variation. Three modes capture 99.9996\% --- for this coefficient representation, nearly all observed variation. Yield-curve PCA also exhibits a concentrated three-factor structure (Litterman and Scheinkman, 1991), but the two studies concern different feature spaces and do not establish a cross-representation performance ranking.

### 3.2 Reconstruction Quality

Table 2 shows the VaR reconstruction error for $r = 3$ modes.

| Statistic | Coefficient $L^2$ error | VaR(5\%) error |
|---|---|---|
| Mean | 0.000385 | 0.005\% |
| Median | 0.000314 | 0.004\% |
| Max | 0.001858 | 0.0416\% |

The maximum relative VaR error from using 3 numbers instead of 128 is 0.0416\% under the declared simulation protocol. This is a finite, source-bound simulation result; no operational materiality threshold is inferred.

### 3.3 Mode Interpretation

**Mode 1** ($z_1$, 97.8\% of variance): the dominant mode is strongly associated with simulated VaR(5\%) ($r = 0.998$). PCA signs are arbitrary, so this supports an association with the simulated risk profile rather than a signed causal interpretation.

**Mode 2** ($z_2$, 2.2\% of variance): a residual mode orthogonal to the leading coordinate. Its economic interpretation is untested.

**Mode 3** ($z_3$, 0.02\% of variance): an extremely small residual mode. Its economic or tail interpretation has not been quantitatively established.

**Remark.** The yield curve analogy is instructive. Three PCA modes of the Treasury yield curve are often interpreted as "level, slope, curvature" (Litterman and Scheinkman, 1991). Our first spectral mode has a strong simulated association with risk level, while the interpretations of the remaining modes as asymmetry and tail structure remain hypotheses for future validation.

### 3.4 Anomaly Detection

The Mahalanobis distance in the 3D $z$-space is a descriptive outlier score only. Under the current uncalibrated $2.5\sigma$ implementation it flags 22 observations in this simulation. This count is not a detection result: no labeled regimes, false-positive control, or out-of-sample detection evaluation is supplied.

### 3.5 Compression Summary

| Quantity | Raw | Compressed | Ratio |
|---|---|---|---|
| Per day | 128 coefficients | 3 numbers | 43$\times$ |
| 250 days | 32,000 numbers | 1,262 numbers | 25$\times$ |
| Information loss | --- | $<$ 0.05\% VaR error | --- |

### 3.6 Out-of-Sample Validation

To test whether the learned basis generalizes beyond its training window, we split the 250-day simulation into a training set (days 1--200) and a holdout set (days 201--250). The SVD is computed on the training period only, yielding principal modes $v_1^{\text{train}}, v_2^{\text{train}}, v_3^{\text{train}}$. We then project the holdout days onto this basis and measure reconstruction quality.

| Metric | Training days 1--200 | Holdout days 201--250 |
|---|---:|---:|
| Mean relative VaR(5\%) error | 0.00191\% | 0.01095\% |
| Max relative VaR(5\%) error | 0.01058\% | 0.01890\% |

The holdout result is a replayable finite simulation diagnostic, not evidence of real-world generalization or of a practical materiality threshold. The exact protocol, source hashes, and values are recorded in `forge/fin_learned_basis/replay_results_20260809.json`.

### 3.7 Robustness Scope

The current reproducibility bundle covers only the declared five-asset simulation and its chronological holdout. Earlier exploratory portfolio-size, correlation-regime, crisis, and raw-return comparisons remain preserved in the repository history, but are not source-bound to the current public protocol and are therefore not claims of this version.

**Deferred portfolio-size analysis.** A future robustness package must expose each scenario, seed, domain rule, and common task metric before it is promoted into the manuscript.

The historical numerical table and accompanying crisis and raw-return comparisons are deliberately withheld here pending that package; their removal is a scope restriction, not deletion of the underlying exploratory artifacts.

---

## 4. The Unification Property

The learned basis is not just a compression tool. It offers a **candidate reduced-order coordinate system** for the spectral risk research program, subject to model-specific approximation validation:

### 4.1 Dynamic spectral update

The temporal compression of spectral coefficients can be studied through PCA. With the learned basis, Dynamic URRT may be approximated by tracking $z_1(t), z_2(t), z_3(t)$ instead of 128 coefficients, subject to validation of the resulting approximation.

### 4.2 Bayesian live-risk approximation

A candidate reduced-order Bayesian posterior is:

$$P(z_1, z_2, z_3 \mid \text{data}).$$

A 3D Kalman filter is a candidate approximation to a 128D filter. The quoted computational reduction by a factor of $(128/3)^3 \approx 77{,}000$ applies only if that approximation is validated for the specified model.

### 4.3 Schrödinger Bridge (Direction 11)

An approximate reduced-order representation of the minimum-entropy path between two distributions is a path in $\mathbb{R}^3$:

$$z(s), \quad s \in [t, t+1]$$

connecting today's $(z_1, z_2, z_3)$ to tomorrow's. Whether this approximation preserves the relevant optimal-transport structure remains to be tested.

### 4.4 Instanton Paths (Direction 13)

A candidate reduced-order crash scenario is a path in $z$-space from the current state to a high-VaR state. The corresponding 3D Euler--Lagrange equations require validation against the original 128D problem.

---

## 5. Relationship to Yield Curve PCA

The closest analogy in finance is the PCA decomposition of interest rate yield curves (Litterman and Scheinkman, 1991; Dai and Singleton, 2000). Both share the same structure:

| Property | Yield curve PCA | Spectral coefficient PCA |
|---|---|---|
| Object | Bond yields $y(T)$ | Spectral coefficients $A_k$ |
| Dimension | $\sim$10 maturities | 128 modes |
| Modes needed | 3 (level, slope, curve) | 3 (empirical simulated coordinates) |
| Variance by Mode 1 | $\sim$85--90\% | 97.8\% |
| Variance by 3 modes | $\sim$99\% | $\sim$100\% |
| Physical interpretation | Interest rate dynamics | Tentative simulated risk-coordinate interpretation |

The simulated spectral coefficient representation exhibits a concentrated explained-variance profile. A fair comparison of compressibility with yield curves would require a common task and evaluation criterion.

---

## 6. The 130 vs 3 Distinction

A natural question arises: if 3 numbers suffice, why do we need 130?

The answer is that the two representations serve fundamentally different purposes:

| Property | 130 parameters (URRT) | 3 parameters (PCA) |
|---|---|---|
| What it encodes | Any distribution, from scratch | Change relative to a known baseline |
| Prerequisites | None (portfolio data only) | 250 days of training + stored basis |
| Guarantee | Mathematical (Theorem 7, Nagy 2026a) | Empirical (conditional on training regime) |
| Failure mode | Scope and assumptions of the cited URRT result | Regime change outside training range |
| Analogy | GPS coordinates (any point on Earth) | "3 blocks left" (requires knowing where you are) |

Under the assumptions and scope of the cited URRT result, the 130-parameter representation is claimed to be universal. This paper does not restate or prove that companion result, and its three-mode compression remains empirical and regime-conditional. In a simulated regime change (2008-type crisis), the learned basis becomes stale and the full 130 parameters may be needed until the basis is re-estimated.

The correct interpretation is hierarchical:

1. Under its stated assumptions, the **URRT** result claims that 130 numbers are sufficient for a single snapshot (static).
2. The **PCA** finds that, in the simulated regime studied here, these 130 numbers move predominantly along 3 observed directions (dynamic).
3. In the simulated regime change, the 3-dimensional trajectory leaves the learned subspace, indicating that the basis may need re-estimation; calibrated anomaly detection remains open.

This hierarchy motivates a bridge to Bayesian Live Risk (Direction #7): the 3 PCA coordinates could serve as an approximate state space for a Kalman filter, but the filtering approximation and any diagnostic based on posterior width require separate validation.

---

## 7. Verification Status and Scope

This paper reports a numerical simulation study. Its empirical percentages, reconstruction errors, and stress illustrations are not machine-verified mathematical theorems. Earlier development notes associated several elementary PCA identities with legacy Lean artifacts, but the current repository snapshot does not provide a source-bound, independently buildable Lean package for the table previously shown here.

The canonical formal source currently registers the relevant learned-basis names as hypotheses or conditions. That is useful dependency metadata, but it is not evidence that Bessel's inequality, Eckart--Young optimality, the VaR reconstruction bound, or the Mahalanobis threshold has been derived end-to-end in the current proof system. Accordingly, this version makes no publication-grade formal-verification claim. A future formal claim requires an exact source path and hash, theorem statements that do not assume their conclusions, a successful kernel replay, and an independently checked Lean export.

## 8. What This Paper Does Not Claim

The study does not establish that real portfolio-loss dynamics are three-dimensional. It does not provide a calibrated anomaly detector, validate Expected Shortfall or other risk functionals, or prove that a projected Bayesian filter or optimal-transport problem preserves the behavior of the original 128-dimensional system. It also does not establish a universal materiality threshold or a publication-grade machine verification of the reported numerical findings.

---

## 9. Conclusion

For the simulated coefficient trajectories studied here, the 128 spectral coordinates are effectively three-dimensional over time: PCA finds that a single mode captures 97.8\% of observed coordinate variation, and three modes capture 99.9996\% (Table 1, Section 3.1). Because the cumulant-based integration domain can vary by day, this is a coordinate-level empirical result conditional on that domain rule, rather than a fixed-function-space conclusion.

The practical implication is a 3-number risk dashboard:

- $z_1$ = a coordinate strongly associated with simulated VaR
- $z_2$ = a residual coordinate with tentative economic interpretation
- $z_3$ = a small residual coordinate with untested tail interpretation

On this simulation, the three coordinates reconstruct VaR(5\%) within the reported error below 0.05\%. Accuracy for ES, other quantiles, moments, and other spectral risk measures remains to be tested.

The learned basis offers a candidate reduced-order representation for the spectral risk program. Dynamic URRT, Bayesian filtering, optimal transport, and instanton analysis require model-specific approximation validation before they can be treated as 3-dimensional problems.

**Limitations.** The reported evidence is a finite simulation with one five-asset protocol and one chronological holdout (Section 3.6). The daily cumulant-based integration domain can vary, so the compression is a coordinate-level analysis conditional on that domain rule rather than a fixed-function-space result. Validation on real market data, Expected Shortfall and other risk functionals, cross-portfolio robustness, fair comparative benchmarks, and calibrated out-of-sample detection remains essential future work. The 3-mode sufficiency is conditional on the training regime being representative; under a structural break, the basis may need re-estimation. No publication-grade formal verification claim is made for these numerical results.

---

## References

- Dai, Q. and Singleton, K. J. (2000). Specification Analysis of Affine Term Structure Models. *The Journal of Finance*, 55(5), 1943-1978. DOI: 10.1111/0022-1082.00278
- Fang, Fang and Oosterlee, Cornelis W. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM Journal on Scientific Computing*, 31(2), 826-848. DOI: 10.1137/080718061
- Litterman, R. and Scheinkman, J (1991). Common factors affecting bond returns. *Litterman, R. and Scheinkman, J.*, 1(1). DOI: 10.3905/jfi.1991.692347
- Nagy, T. (2026). Deterministic Log-CF Evaluation for Correlated Lognormal Sums. *Zenodo*. DOI: 10.5281/zenodo.21443048
- Nagy, T. (2026). Exact Portfolio VaR Without Monte Carlo: The Eigen-COS Method. *Zenodo*. DOI: 10.5281/zenodo.18910516
- Nagy, T. (2026). Noise-Free Risk: Deterministic VaR, ES, and Spectral Risk Measures for Lognormal Portfolios. *Working paper*.
- Nagy, T. (2026). The Anomaly Functional: Real-Time Arbitrage Detection via Spectral Risk Coefficients. *Working paper*.
