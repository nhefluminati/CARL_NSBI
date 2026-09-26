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

import numpy as np

from .combine import DEFAULT_COMBINER, validate_combiner
from .config import RunRecord, load_config
from .data.dataset import NSBIDataset, SplitIndices
from .data.loading import DatasetBuilder, save_reference_cache
from .data.reweighting import ReweightStep
from .data.scaling import StandardScalerStep
from .data.splitting import (
    DEFAULT_FOLD_SEED,
    TEST_FOLD,
    KFoldStep,
    SplitStep,
    reference_fingerprint,
)
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
            reference_paths=data_cfg.get("reference_paths", []) or [],
            features=data_cfg["features"],
            weight_key=data_cfg.get("weight_key", "weight"),
            absolute_weights=data_cfg.get("absolute_weights", False),
            load_reference=data_cfg.get("load_reference"),
            save_reference=data_cfg.get("save_reference"),
        )
        split_cfg = config.get("split", {})
        self.split_step = SplitStep(
            train_fraction=split_cfg.get("train_fraction", 0.8),
            val_fraction=split_cfg.get("val_fraction", 0.1),
            seed=self.seed,
        )
        # k-fold cross-validation replaces the fixed train/val/test split and
        # the bootstrap pooling for ensemble training (see KFoldStep).
        k_folds = int(split_cfg.get("k_folds", 0) or 0)
        self.kfold_step: KFoldStep | None = (
            KFoldStep(
                k=k_folds,
                fold_seed=int(split_cfg.get("fold_seed", DEFAULT_FOLD_SEED)),
                val_fraction=float(split_cfg.get("kfold_val_fraction", 0.2)),
                test_fraction=float(split_cfg.get("kfold_test_fraction", 0.1)),
            )
            if k_folds
            else None
        )
        self.reweight_step = ReweightStep(
            reference_unit_weights=data_cfg.get("reference_unit_weights", True),
            normalize_weights=data_cfg.get("normalize_weights", False),
            target_balance_factor=data_cfg.get("target_balance_factor", 1.0),
        )
        self.scaler_step = StandardScalerStep()

        model_cfg = config.get("model", {})
        sched_cfg = model_cfg.get("scheduler", {})
        train_cfg = config.get("training", {})
        perf_cfg = config.get("performance", {})
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
                optimizer=train_cfg.get("optimizer", "sgd"),
                weight_decay=train_cfg.get("weight_decay", 0.0),
                precision=perf_cfg.get("precision", "32-true"),
                device_batches=perf_cfg.get("device_batches", True),
                matmul_precision=perf_cfg.get("matmul_precision", "high"),
                compile=perf_cfg.get("compile", False),
                save_last=perf_cfg.get("save_last", False),
                checkpoint_every_n_epochs=perf_cfg.get("checkpoint_every_n_epochs", 1),
                progress_bar=perf_cfg.get("progress_bar", True),
                log_weights=perf_cfg.get("log_weights", True),
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
                    bootstrap_reference=ens_cfg.get("bootstrap_reference", False),
                    workers_per_gpu=ens_cfg.get("workers_per_gpu", 1),
                    start_member=ens_cfg.get("start_member", 0),
                    mode=ens_cfg.get("mode", "vectorized"),
                    members_per_group=ens_cfg.get("members_per_group", 0),
                    combiner=validate_combiner(
                        ens_cfg.get("combiner", DEFAULT_COMBINER),
                        ens_cfg.get("combiner_trim", 0.1),
                    ),
                    combiner_trim=float(ens_cfg.get("combiner_trim", 0.1)),
                    k_folds=self.kfold_step.k if self.kfold_step else 0,
                    fold_seed=self.kfold_step.fold_seed if self.kfold_step else DEFAULT_FOLD_SEED,
                    kfold_val_fraction=(
                        self.kfold_step.val_fraction if self.kfold_step else 0.2
                    ),
                    kfold_test_fraction=(
                        self.kfold_step.test_fraction if self.kfold_step else 0.1
                    ),
                ),
                output_dir=self.output_dir,
                run_name=self.run_name,
            )

        if self.kfold_step is not None and self.ensemble is None:
            raise ValueError(
                "split.k_folds needs ensemble.enabled: true -- k-fold trains one ensemble "
                "per fold (ensemble.n_members members each)."
            )

        # filled by run():
        self.dataset: NSBIDataset | None = None
        self.splits: SplitIndices | None = None
        self.folds: np.ndarray | None = None

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

        # The fixed split always runs: it assigns the per-event split labels a
        # reference cache stores, so caches stay usable in both modes.
        splits = self.split_step.split(dataset)
        if self.kfold_step is None:
            self.record.update(
                split={
                    "mode": "fixed",
                    "train_fraction": self.split_step.train_fraction,
                    "val_fraction": self.split_step.val_fraction,
                    "n_train": len(splits.train),
                    "n_val": len(splits.val),
                    "n_test": len(splits.test),
                }
            )
        else:
            # k-fold: the cross-validated events (folds 0..k-1) are used for
            # training (in k-1 of the folds' ensembles) and scored by their own
            # fold's. The untouched test set (fold -1) is kept out of all of it
            # -- including the reweighting and scaler fits below, which are
            # fixed on the cross-validated events only. Each member restores
            # its own 1:1 class balance on the rows it actually trains on.
            self.folds = self.kfold_step.assign(dataset)
            cv = np.flatnonzero(self.folds != TEST_FOLD).astype(np.int64)
            splits = SplitIndices(
                train=cv,
                val=np.empty(0, np.int64),
                test=np.flatnonzero(self.folds == TEST_FOLD).astype(np.int64),
            )
            self.record.update(
                split={
                    "mode": "kfold",
                    "k_folds": self.kfold_step.k,
                    "fold_seed": self.kfold_step.fold_seed,
                    "kfold_val_fraction": self.kfold_step.val_fraction,
                    "kfold_test_fraction": self.kfold_step.test_fraction,
                    "n_per_fold": np.bincount(
                        self.folds[cv], minlength=self.kfold_step.k
                    ).tolist(),
                    "n_test": len(splits.test),
                }
            )

        # The reference is the common denominator of every template's network,
        # so record a hash of it. Two runs agreeing on these digests trained
        # against byte-identical reference events.
        self.record.update(reference_fingerprint=reference_fingerprint(dataset, splits))

        # Saved before reweighting: the stored weights are the ones read from
        # the inputs (with |w| applied if configured). The reweighting is
        # relative to the target yield and so is recomputed every run.
        if self.dataset_builder.save_reference:
            self.record.update(
                reference_cache_written=save_reference_cache(
                    self.dataset_builder.save_reference,
                    dataset,
                    self.dataset_builder.absolute_weights,
                )
            )
        if self.dataset_builder.load_reference:
            self.record.update(reference_cache_loaded=self.dataset_builder.load_reference)

        self.record.update(reweighting=self.reweight_step.apply(dataset, splits))
        self.record.update(preprocessing=self.scaler_step.apply(dataset, splits))

        self.dataset, self.splits = dataset, splits
        return dataset, splits

    def run(self) -> dict:
        dataset, splits = self.prepare_data()

        if self.ensemble is not None and self.kfold_step is not None:
            summaries = self.ensemble.train(dataset, splits, folds=self.folds)
            k = self.kfold_step.k
            by_fold = [[s["tag"] for s in summaries if s.get("fold") == f] for f in range(k)]
            # Written before evaluation so a crash there still leaves a
            # usable record behind.
            self.record.update(
                ensemble_members=[s["tag"] for s in summaries],
                kfold={
                    "k_folds": k,
                    "fold_seed": self.kfold_step.fold_seed,
                    "kfold_val_fraction": self.kfold_step.val_fraction,
                    "kfold_test_fraction": self.kfold_step.test_fraction,
                    "members": by_fold,
                },
            )
            metrics = self.ensemble.evaluate(
                dataset, np.arange(len(dataset)), tag="ensemble_test", folds=self.folds
            )
            self.record.update(final_metrics=metrics)
            return {"summaries": summaries, "metrics": metrics, "ensemble": self.ensemble,
                    "folds": self.folds}

        if self.ensemble is not None:
            summaries = self.ensemble.train(dataset, splits)
            metrics = self.ensemble.evaluate(dataset, splits.test)
            self.record.update(ensemble_members=[s["tag"] for s in summaries], final_metrics=metrics)
            return {"summaries": summaries, "metrics": metrics, "ensemble": self.ensemble}

        summary = self.trainer.fit(dataset, splits, seed=self.seed)
        self.record.update(final_metrics_file=str(self.output_dir / "evaluation"))
        return {"summaries": [summary]}
