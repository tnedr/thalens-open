"""
Latent Compiler — command-line interface.

Examples:
    # Compile a built-in Boolean demo map exactly.
    python -m latent_compiler boolean --map rotate_left --m-bits 10

    # Compile every built-in Boolean demo map at 1% tolerance.
    python -m latent_compiler boolean --map all --m-bits 10 --tolerance 0.01

    # Compile a linear operator saved as a 2D array in an .npz under key 'A'.
    python -m latent_compiler matrix --input operator.npz --key A --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import latent_compile, walsh_koopman_operator
from .lowering import round_trip_error


# Minimal built-in demo maps. The full experimental zoo lives in
# experiments/algorithmic_latent_bit_dynamics.py; these are just for the CLI.
def _rotate_left(x: np.ndarray, m: int) -> np.ndarray:
    mask = (1 << m) - 1
    return ((x << 1) | (x >> (m - 1))) & mask


def _lfsr(x: np.ndarray, m: int) -> np.ndarray:
    mask = (1 << m) - 1
    taps = (1 << (m - 1)) | (1 << 1) | 1
    return np.where((x & 1) == 1, (x >> 1) ^ taps, x >> 1) & mask


def _rule30(x: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(m):
        left = (x >> ((i - 1) % m)) & 1
        center = (x >> i) & 1
        right = (x >> ((i + 1) % m)) & 1
        out |= (left ^ (center | right)) << i
    return out


def _random_permutation(x: np.ndarray, m: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.permutation(1 << m)[x]


BUILTIN_MAPS = {
    "rotate_left": _rotate_left,
    "lfsr": _lfsr,
    "rule30": _rule30,
    "random_permutation": _random_permutation,
}


def _result_to_dict(name: str, result) -> dict:
    rep = {k: v for k, v in result.representation.items() if not isinstance(v, np.ndarray)}
    return {
        "name": name,
        "accepted": result.accepted,
        "basis": result.basis,
        "dimension": result.dimension,
        "tolerance": result.tolerance,
        "latent_complexity": result.latent_complexity,
        "certificate": {
            "kind": result.certificate.kind,
            "relative_error": result.certificate.relative_error,
            "verified_on": result.certificate.verified_on,
            "holds": result.certificate.holds,
        },
        "cost": {
            "ambient_units": result.cost.ambient_units,
            "latent_units": result.cost.latent_units,
            "ideal_speedup": result.cost.ideal_speedup,
            "beneficial": result.cost.beneficial,
            "break_even_calls": result.cost.break_even_calls,
        },
        "reason": result.reason,
        "representation": rep,
    }


def _run_boolean(args) -> dict:
    m = args.m_bits
    states = np.arange(1 << m, dtype=np.int64)
    names = list(BUILTIN_MAPS) if args.map == "all" else [args.map]
    if any(name not in BUILTIN_MAPS for name in names):
        available = ", ".join(BUILTIN_MAPS)
        raise SystemExit(f"unknown --map {args.map!r}; available: {available}, all")

    out = []
    for name in names:
        truth_table = np.asarray(BUILTIN_MAPS[name](states, m), dtype=np.int64) & ((1 << m) - 1)
        result = latent_compile(truth_table, m_bits=m, tolerance=args.tolerance)
        rt = round_trip_error(result, walsh_koopman_operator(truth_table, m))
        record = _result_to_dict(name, result)
        record["round_trip_error"] = rt
        out.append(record)
    return {"mode": "boolean", "m_bits": m, "tolerance": args.tolerance, "results": out}


def _run_matrix(args) -> dict:
    data = np.load(args.input)
    A = np.asarray(data[args.key], dtype=float)
    result = latent_compile(A, tolerance=args.tolerance, value_width=args.value_width)
    rt = round_trip_error(result, A)
    record = _result_to_dict(Path(args.input).stem, result)
    record["round_trip_error"] = rt
    return {"mode": "matrix", "tolerance": args.tolerance, "results": [record]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Certifying latent compiler")
    sub = parser.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("boolean", help="compile a Boolean map")
    pb.add_argument("--map", default="all", help="built-in map name or 'all'")
    pb.add_argument("--m-bits", type=int, default=10)
    pb.add_argument("--tolerance", type=float, default=0.0)
    pb.add_argument("--output", default=None)

    pm = sub.add_parser("matrix", help="compile a linear operator from an .npz")
    pm.add_argument("--input", required=True)
    pm.add_argument("--key", default="A")
    pm.add_argument("--tolerance", type=float, default=0.0)
    pm.add_argument("--value-width", type=int, default=1)
    pm.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    payload = _run_boolean(args) if args.mode == "boolean" else _run_matrix(args)

    for record in payload["results"]:
        verdict = "COMPILED" if record["accepted"] else "REJECTED"
        print(
            f"{record['name']:<22}{verdict:>10}"
            f"  err={record['certificate']['relative_error']:.2e}"
            f"  speedup={record['cost']['ideal_speedup']:.1f}x"
            f"  round_trip={record['round_trip_error']:.2e}"
            f"  :: {record['reason']}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
