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
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
