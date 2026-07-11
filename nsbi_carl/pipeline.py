"""The pipeline masterclass.

Composes the step objects — dataset construction, splitting, reweighting,
preprocessing, training (single network or ensemble), evaluation — and
executes them in order. Everything is configured from one YAML file::

    Pipeline.from_yaml("configs/example.yaml").run()

At the start of the run a run-record YAML is created in the output directory
and each step appends what it fixed (feature order, scaler mean/std,
reweighting scale factors, seeds, ...).
"""

from __future__ import annotations

from pathlib import Path

from .config import RunRecord, load_config
from .data.dataset import NSBIDataset, SplitIndices
from .data.loading import DatasetBuilder
from .data.reweighting import ReweightStep
from .data.scaling import StandardScalerStep
from .data.splitting import SplitStep
from .evaluation import base as _eval_base
from .evaluation import metrics as _metrics  # noqa: F401  (populate registries)
from .evaluation import plots as _plots      # noqa: F401
from .training.ensemble import CARLEnsemble, EnsembleConfig
from .training.trainer import CARLTrainer, ModelConfig, TrainerConfig


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.run_name: str = config.get("run_name", "carl")
        self.output_dir = Path(config.get("output_dir", f"outputs/{self.run_name}"))
        self.seed: int = int(config.get("seed", 52))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record = RunRecord(self.output_dir / f"run_config_{self.run_name}.yaml")

        # ---- step construction (composition over inheritance) ------------
        data_cfg = config["data"]
        self.dataset_builder = DatasetBuilder(
            target_paths=data_cfg["target_paths"],
            reference_paths=data_cfg["reference_paths"],
            features=data_cfg["features"],
            weight_key=data_cfg.get("weight_key", "weight"),
        )
        split_cfg = config.get("split", {})
        self.split_step = SplitStep(
            train_fraction=split_cfg.get("train_fraction", 0.8),
            val_fraction=split_cfg.get("val_fraction", 0.1),
            seed=self.seed,
        )
        self.reweight_step = ReweightStep()
        self.scaler_step = StandardScalerStep()

        model_cfg = config.get("model", {})
        sched_cfg = model_cfg.get("scheduler", {})
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        self.trainer = CARLTrainer(
            model_config=ModelConfig(
                n_layers=model_cfg.get("n_layers", 5),
                n_nodes=model_cfg.get("n_nodes", 128),
                dropout=model_cfg.get("dropout", 0.0),
                scheduler_t0=sched_cfg.get("t0", 25),
                scheduler_t_mult=sched_cfg.get("t_mult", 1),
                scheduler_eta_min=sched_cfg.get("eta_min", 1e-8),
            ),
            trainer_config=TrainerConfig(
                batch_size=train_cfg.get("batch_size", 512),
                val_batch_size=train_cfg.get("val_batch_size", 2048),
                learning_rate=train_cfg.get("learning_rate", 1e-4),
                momentum=train_cfg.get("momentum", 0.0),
                max_epochs=train_cfg.get("max_epochs", 400),
                early_stopping_patience=train_cfg.get("early_stopping_patience", 25),
                num_workers=train_cfg.get("num_workers", 4),
                gpus=train_cfg.get("gpus", []) or [],
                resume_from=train_cfg.get("resume_from"),
            ),
            output_dir=self.output_dir,
            run_name=self.run_name,
            training_evaluation=_eval_base.build_suite(eval_cfg.get("during_training")),
            final_evaluation=_eval_base.build_suite(eval_cfg.get("after_training")),
        )

        ens_cfg = config.get("ensemble", {})
        self.ensemble: CARLEnsemble | None = None
        if ens_cfg.get("enabled", False):
            self.ensemble = CARLEnsemble(
                trainer=self.trainer,
                ensemble_config=EnsembleConfig(
                    n_members=ens_cfg.get("n_members", 8),
                    start_seed=ens_cfg.get("start_seed", self.seed),
                    bootstrap=ens_cfg.get("bootstrap", True),
                    bootstrap_fraction=ens_cfg.get("bootstrap_fraction", 1.0),
                    workers_per_gpu=ens_cfg.get("workers_per_gpu", 1),
                    start_member=ens_cfg.get("start_member", 0),
                ),
                output_dir=self.output_dir,
                run_name=self.run_name,
            )

        # filled by run():
        self.dataset: NSBIDataset | None = None
        self.splits: SplitIndices | None = None

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        return cls(load_config(path))

    # ------------------------------------------------------------------
    def prepare_data(self) -> tuple[NSBIDataset, SplitIndices]:
        """Steps 1-4: build -> split -> reweight -> scale (in that order)."""
        self.record.update(run_name=self.run_name, seed=self.seed, config=self.config)

        dataset = self.dataset_builder.build()
        self.record.update(
            features=dataset.feature_names,
            samples=dataset.sample_names,
            n_target=dataset.n_target,
            n_reference=dataset.n_reference,
        )

        splits = self.split_step.split(dataset)
        self.record.update(
            split={
                "train_fraction": self.split_step.train_fraction,
                "val_fraction": self.split_step.val_fraction,
                "n_train": len(splits.train),
                "n_val": len(splits.val),
                "n_test": len(splits.test),
            }
        )

        self.record.update(reweighting=self.reweight_step.apply(dataset, splits))
        self.record.update(preprocessing=self.scaler_step.apply(dataset, splits))

        self.dataset, self.splits = dataset, splits
        return dataset, splits

    def run(self) -> dict:
        dataset, splits = self.prepare_data()

        if self.ensemble is not None:
            summaries = self.ensemble.train(dataset, splits)
            metrics = self.ensemble.evaluate(dataset, splits.test)
            self.record.update(ensemble_members=[s["tag"] for s in summaries], final_metrics=metrics)
            return {"summaries": summaries, "metrics": metrics, "ensemble": self.ensemble}

        summary = self.trainer.fit(dataset, splits, seed=self.seed)
        self.record.update(final_metrics_file=str(self.output_dir / "evaluation"))
        return {"summaries": [summary]}
