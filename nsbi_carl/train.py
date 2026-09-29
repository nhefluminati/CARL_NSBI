"""CLI entrypoint:  python -m nsbi_carl.train --config configs/example.yaml"""

from __future__ import annotations

import argparse

from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CARL/NSBI training pipeline.")
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config.")
    args = parser.parse_args()

    result = Pipeline.from_yaml(args.config).run()
    if "metrics" in result:
        print("Final ensemble metrics:")
        _print_metrics(result["metrics"])


def _print_metrics(metrics: dict, indent: str = "  ") -> None:
    """Flat for a fixed-split run; nested {"test": .., "out_of_fold": ..} for k-fold."""
    for k, v in (metrics or {}).items():
        if isinstance(v, dict):
            print(f"{indent}{k}:")
            _print_metrics(v, indent + "  ")
        elif isinstance(v, (int, float)):
            print(f"{indent}{k}: {v:.6f}")
        else:
            print(f"{indent}{k}: {v}")


if __name__ == "__main__":
    main()
