# nsbi_carl

A reworked, composable training workspace for CARL/NSBI classifier ensembles
(PyTorch Lightning). One YAML config drives the whole pipeline; every
responsibility lives in its own class; nothing is hardcoded in scripts.

```
python -m nsbi_carl.train --config configs/example.yaml
```

## Architecture

```
Pipeline (masterclass, built from YAML)
 ├─ DatasetBuilder        reads target/reference .h5 → NSBIDataset (labels 1/0)
 ├─ SplitStep             deterministic per-sample train/val/test split
 ├─ ReweightStep          (1) equalize reference samples (2) balance target vs reference
 ├─ StandardScalerStep    mean-0 / var-1, fitted on train only
 ├─ CARLTrainer           one network: loaders, lightning.Trainer, early stopping,
 │                        checkpoints/resume, multi-GPU, evaluation suites
 └─ CARLEnsemble          pools members from the common dataset, parallel training,
                          .inference() = mean of member predictions,
                          ensemble-level evaluation phase
```

Steps run in exactly this order. A run record
(`<output_dir>/run_config_<name>.yaml`) is created at the start and every step
writes what it fixed: feature list **and order**, scaler mean/std,
per-reference-sample scales, the target balance scale, split sizes, seeds.

## NSBI guarantees baked into the data steps

- Target events → label 1, reference events → label 0.
- Each reference sample is normalized to identical total weight **before**
  the target/reference balancing.
- Target weights are multiplied by
  `w_reference_train.sum() / w_target_train.sum()` (train split only!), and
  that *numerically identical* scalar is applied to train, validation and
  test — so val/test use the exact same reweighting as train.
- The split is drawn per sample with a generator seeded by
  `(seed, crc32(sample_name))`: for the same seed, the same events are pulled
  from a reference sample regardless of which target samples are present.
  Ensemble bootstraps resample target/reference with separate generators for
  the same reason.
- The scaler is fitted on the train split only and applied everywhere.

## Model

`CARL(LightningModule)`: configurable `n_layers` / `n_nodes` / `dropout`,
swish (SiLU) hidden activations, single sigmoid output, weighted BCE loss,
SGD optimizer, `CosineAnnealingWarmRestarts(T_0, T_mult, eta_min)` scheduler.
Train/val loss are logged each epoch.

## Metrics & plots

Metrics implement `.evaluate(ctx)`, plots implement `.plot(ctx)`. They are
registered by name and configured in YAML — add a new diagnostic without
touching any script:

```python
from nsbi_carl import Metric, register_metric

@register_metric("my_metric")
class MyMetric(Metric):
    def evaluate(self, ctx):
        return {"my_metric": float((ctx.ratio[ctx.labels == 0]).mean())}
```

Both receive an `EvaluationContext` (scores, labels, weights, features,
scaler, histories, optional `member_scores`), which is what makes every
metric/plot work identically for a single network during training, after
training, and for the entire ensemble.

Included: `density_ratio_integral` (Σ_ref r·w/Σw and its deviation from 1),
`loss_curves`, `calibration_curve`, `reweighting` (closure plot with ratio
panel, per feature).

They are trivially invocable on the fly, e.g. on any test dataset:

```python
from nsbi_carl import EvaluationContext
from nsbi_carl.evaluation.plots import CalibrationCurvePlot

CalibrationCurvePlot(n_bins=40).plot(EvaluationContext(scores=s, labels=y, weights=w))
```

## Ensembles

`CARLEnsemble` reuses `CARLTrainer` per member, trains members in parallel
(round-robin over `training.gpus`, `workers_per_gpu` each, spawn-safe via an
.npz dataset snapshot), with member `i` seeded `start_seed + i`. Resume /
extend with `start_member`. Afterwards:

```python
result = Pipeline.from_yaml("configs/example.yaml").run()
ensemble = result["ensemble"]
scores = ensemble.inference(x_scaled)   # average over members
```

The final evaluation phase runs once for the whole ensemble on the test
split; contexts then also carry the full `(n_members, n_events)` prediction
matrix.

## Programmatic use

```python
from nsbi_carl import Pipeline

pipe = Pipeline.from_yaml("configs/example.yaml")
dataset, splits = pipe.prepare_data()   # steps 1-4 only
# ... or pipe.run() for everything
```
