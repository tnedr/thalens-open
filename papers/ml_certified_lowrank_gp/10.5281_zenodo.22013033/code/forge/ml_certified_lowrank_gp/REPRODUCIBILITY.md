# Reproducing the Certified Low-Rank GP Results

This archive accompanies version 0.2 of *Certified Low-Rank Gaussian Processes
via the Noise-Floor Rule*. It preserves the source layout expected by the
benchmark runner and contains the exact JSON receipt used by the manuscript.

## Requirements

- Python 3.10 or newer;
- NumPy 1.24 or newer;
- SciPy 1.10 or newer.

The certified build used Python 3.12.3, NumPy 2.5.0, and SciPy 1.18.0.

## Evidence scope

The current source-bound positive-case receipt is
`forge/ml_certified_lowrank_gp/reproduction_0813_full.json`. It records the
smooth RBF experiment used for the manuscript's accepted low-rank branch.

The archive also contains
`forge/meta_algorithmic_latent/latent_gp_certified_results.json` as historical
rough-control evidence. Its `rough_rbf_control` case uses 4,000 observations,
length scale 0.25, and noise variance 0.01. The discovered rank is 4,000, the
candidate is not accepted, and the recorded verdict is
`rejected_fall_back_to_exact`. The older smooth-case timing and certificate
fields in that historical file are not the current manuscript receipt and
must not be interpreted as such.

## Fast non-overwriting replay

From the extracted archive root, run:

```text
python forge/ml_certified_lowrank_gp/smoke_reproduction.py
```

The command asserts that the low-rank factor is accepted, the certificate is
bound to the solver factor through the Frobenius triangle bound, and the actual
solve error lies below the certified bound. It does not overwrite any receipt.

## Full source-bound replay

Run:

```text
python forge/ml_certified_lowrank_gp/reproduction_0813_full.py
```

This executes the manuscript protocol with 4,000 observations and writes
`forge/ml_certified_lowrank_gp/reproduction_0813_full.json`. On the original
host, discovery took approximately 631 seconds. The archive already contains
the immutable receipt used by the paper so it can be inspected before choosing
to rerun the expensive protocol.

## Release identity

- Manuscript source SHA-256:
  `a057fb11e3841cbf80b25e9d410559a829087c8e6b44e43c37dc248cf932d069`
- PDF SHA-256:
  `16c1d1ad7c1edc51f01779941254b68ec63b716e420e5dc3b30e121554a6ca55`
- Full receipt:
  `forge/ml_certified_lowrank_gp/reproduction_0813_full.json`

The DOI and archive SHA-256 are recorded by the publication receipt after the
Zenodo release is verified.
