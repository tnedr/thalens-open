"""Full positive-branch reproduction after solver-factor certificate repair."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import latent_gp_certified_demo as demo


case = demo.run_case(
    "smooth_rbf_lowrank_full", n=4000, d=4, lengthscale=2.5, sigma2=5e-2,
    eps=0.05, method="sketched", max_rank=512, seed=0,
)
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps({"protocol": "full positive-branch certificate reproduction", "case": case}, indent=2), encoding="utf-8")
print(case["label"], case["verdict"], case["certified"].get("certificate_basis"))
print(f"Wrote {output}")
