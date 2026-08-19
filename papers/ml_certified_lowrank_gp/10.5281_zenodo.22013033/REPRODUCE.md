# Reproducing the certified low-rank GP results

The canonical manuscript bytes are preserved in `paper.pdf` and `source/`.
The exact Zenodo reproducibility archive is unpacked under `code/` without
modification. It requires Python 3.10+, NumPy 1.24+, and SciPy 1.10+.

## Fast verified replay

From this directory, run:

```text
cd code
python forge/ml_certified_lowrank_gp/smoke_reproduction.py
```

The publication staging check returned:

```text
verdict: compiled
rank: 142
certificate_basis: solver_factor_frobenius_triangle_bound
alpha_rel_actual_error: 0.0019591720203654436
alpha_rel_certified_bound: 0.04885306753838673
alpha_bound_holds: true
```

This fast replay is non-overwriting. The full 4,000-observation replay is:

```text
cd code
python forge/ml_certified_lowrank_gp/reproduction_0813_full.py
```

The full run is expected to be slow; the archived source-bound receipt is
`code/forge/ml_certified_lowrank_gp/reproduction_0813_full.json`. Historical
rough-control evidence is separately labelled at
`code/forge/meta_algorithmic_latent/latent_gp_certified_results.json`.

Release identity:

- Version DOI: `10.5281/zenodo.22013033`
- Concept DOI: `10.5281/zenodo.22013032`
- Source SHA-256: `a057fb11e3841cbf80b25e9d410559a829087c8e6b44e43c37dc248cf932d069`
- PDF SHA-256: `16c1d1ad7c1edc51f01779941254b68ec63b716e420e5dc3b30e121554a6ca55`
- Zenodo archive SHA-256: `7ac7f0256425df8cac07c80bf432790f4e5afb95cafc4791c8b5d4742ce7a5b7`
