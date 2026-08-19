"""Fast, non-overwriting replay for the certified low-rank GP release."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import latent_gp_certified_demo as demo  # noqa: E402


case = demo.run_case(
    "release_smoke",
    n=800,
    d=4,
    lengthscale=2.5,
    sigma2=5e-2,
    eps=0.05,
    method="sketched",
    max_rank=256,
    seed=0,
)

assert case["verdict"] == "compiled"
assert case["alpha_bound_holds"] is True
assert case["certified"]["accepted"] is True
assert case["certified"]["certificate_basis"] == "solver_factor_frobenius_triangle_bound"

print(json.dumps({
    "verdict": case["verdict"],
    "rank": case["certified"]["rank"],
    "certificate_basis": case["certified"]["certificate_basis"],
    "alpha_rel_actual_error": case["alpha_rel_actual_error"],
    "alpha_rel_certified_bound": case["alpha_rel_certified_bound"],
    "alpha_bound_holds": case["alpha_bound_holds"],
}, indent=2))
