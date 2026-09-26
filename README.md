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
- The scaler is fitted on the train split only and applied everywhere.

## Keeping the reference sample fixed

Every template network (S, B, SBI, qq, SBI_EW) is trained against the same
reference, so the likelihood works with ratios of their outputs. Any
difference in the reference between two of those trainings biases those
ratios directly. Four mechanisms keep it pinned:

1. **Split independence.** The per-sample draw depends only on the sample's
   own name, its own event count and the seed — never on which target is in
   the run, or how many events the target has.
2. **Target-only bootstrap.** `ensemble.bootstrap_reference: false` (the
   default) resamples only the target per member; every member trains on the
   identical, complete reference train split, in a fixed order. The ensemble
   spread then reflects target statistics, which is what it is meant to
   measure. Set it to `true` for the old behaviour.
3. **`data.save_reference`.** Writes the constructed reference events *and
   their train/val/test assignment* to one `.h5`. Written after splitting but
   before reweighting, since the reweighting is relative to the target yield
   and must be recomputed per run.
4. **`data.load_reference`.** Builds the reference from that file instead of
   from `reference_paths`, keeping it fixed even if the seed, the split
   fractions or the reference inputs change. A mismatched feature list (or
   merely a permuted one) is rejected rather than silently applied.

The intended workflow is to build the reference once, then point every
template's config at it:

```yaml
# build_reference.yaml — run once
data: {save_reference: data/reference_sample.h5, ...}

# S.yaml, B.yaml, SBI.yaml, qq.yaml, SBI_EW.yaml
data: {load_reference: data/reference_sample.h5, reference_paths: [], ...}
```

Each run records a `reference_fingerprint` in its run record: a hash of the
actual reference events per split, independent of shuffle order and of how
many target events sit ahead of them. Two runs agreeing on those digests
trained against byte-identical reference data — which makes the guarantee
checkable after the fact rather than merely intended.

### Weights

`data.absolute_weights: true` replaces the reference weights with `|w|`, and
only the reference — target weights keep their sign. Negative MC weights
enter a weighted BCE with the wrong sign and can drive the loss unbounded
below.

`data.reference_unit_weights` (default `true`) overwrites every reference
weight with 1 before the samples are equalized, building a synthetic
reference with the desired domain. **While it is on, `absolute_weights` has
no observable effect**, because the weights it would fix are discarded
immediately afterwards. Turn it off to train against the reference sample's
own weights.

`tests/reference_test.py` checks all of the above.

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

`CARLEnsemble` trains the members in one of two modes, set by `ensemble.mode`.

**`vectorized` (default).** All members assigned to a GPU are stacked into a
single module whose weights carry a leading member axis, so one batched
matmul trains all of them:

```
x (M, B, F)  @  W (M, F, H)  ->  (M, B, H)
```

Members stay mathematically independent — their parameters are disjoint
slices, so summing the per-member losses gives each exactly the gradient it
would have had alone — and member `i` still initialises from `start_seed + i`
and still trains on its own bootstrap resample. Each member gets its own
early-stopping counter and its own best-weight snapshot, and is written out as
an ordinary Lightning checkpoint that `CARL.load_from_checkpoint` reads.

This matters because a CARL network here is tiny. On a modern GPU a 5x256 net
on a handful of features cannot fill the device, so wall time is dominated by
kernel-launch overhead rather than arithmetic. Stacking raises the work per
launch by a factor M while leaving the launch count unchanged.

**`process`.** The original path: one member per process, round-robin over
`training.gpus` with `workers_per_gpu` each. Kept as a fallback.

With multiple GPUs the members are split across them and each GPU trains its
group vectorized, which is strictly better than DDP for this workload — DDP
parallelises a single tiny network, and the gradient all-reduce costs more
than the gradient it synchronises.

Resume / extend with `start_member`. Afterwards:

```python
result = Pipeline.from_yaml("configs/example.yaml").run()
ensemble = result["ensemble"]
scores = ensemble.inference(x_scaled)   # combined per ensemble.combiner
```

**k-fold cross-validation.** `split.k_folds: 10` switches the ensemble from
bootstrap pooling to the scheme of the INT note (Sec. 2.7.2). Every event of
every sample is put in one of k folds; fold f gets `ensemble.n_members`
members, each trained on its own train/validation split of the other k-1
folds, drawn **without** replacement (`split.kfold_val_fraction` to
validation). The bootstrap settings are ignored. Target rows are redrawn per
member; reference rows are fixed per fold, so every member of every template
shares the same reference events within a fold.

Before the folds are dealt, `split.kfold_test_fraction` (default 0.1) of
every sample is set aside as fold -1: a final test set that no member of any
fold trains or validates on, and that the reweighting and scaler fits do not
see either. It is scored by the whole ensemble.

The fold of an event depends only on `split.fold_seed`, the sample's file
name, `k` and the event's row in that file — not on the run seed or on
whether the sample is a target or part of the reference. Use the same
`k_folds`, `fold_seed` and `kfold_test_fraction` for all templates
(`validate_templates` checks it); a reference event then sits in the same fold
— and the same test set — for every template.

Fold f's members are the only networks allowed to score fold f's events;
test-set events are scored by all members:

```python
rec = read_record(run_dir, "S")
scorer = EnsembleScorer(rec)
folds = rec.folds_for("4l_SR_S.h5", n_rows)          # rows in file order
r_S = scorer.score_out_of_fold(x_S, folds)            # all MC: Asimov, closure
x_ref, w_ref, f_ref = load_reference_sample(cache, rec, split="all", return_folds=True)
# split="test": only the untouched test events; split="cv": the rest
r_ref = scorer.score_out_of_fold(x_ref, f_ref)        # reference pool
r_data = scorer.score(x_data)                         # data: all k*n members
```

The after-training evaluation runs twice: `ensemble_test` on the untouched
test set with the whole ensemble, and `ensemble_test_out_of_fold` on the
cross-validated events, each scored by its own fold.

**Combining members.** `ensemble.combiner` sets how the member scores become
one score: `mean_score` (default, the historical behaviour), `mean_ratio`,
`mean_logit` (geometric mean of the ratios), `median_ratio`, or
`trimmed_ratio` (with `combiner_trim`). Because `r = s/(1-s)` is convex,
averaging scores gives a smaller ratio than averaging ratios wherever the
members disagree, which is typically where the reference is thin. The choice
is stored in the run record and picked up by `EnsembleScorer`; override it at
analysis time with `EnsembleScorer(record, combiner="mean_ratio")` or
`scorer.set_combiner(...)` — no retraining needed. See `nsbi_carl/combine.py`.

Inference stacks the members too, so the likelihood scan evaluates all of them
in one pass over the events instead of M sequential sweeps.

The final evaluation phase runs once for the whole ensemble on the test
split; contexts then also carry the full `(n_members, n_events)` prediction
matrix.

## Performance

The `performance:` block in the config holds the knobs; the defaults are the
fast ones. The changes that matter, in order of size:

| Change | Why |
| --- | --- |
| `device_batches: true` | The splits are uploaded to the GPU once and batched by tensor slicing. The default `DataLoader(Subset(...))` path makes one Python `__getitem__` call per **event** per epoch, then collates, then (with workers) pickles each batch through a queue — for a 10M-event split that is ~10M Python calls per epoch to feed two matmuls. |
| `ensemble.mode: vectorized` | M members per kernel launch instead of one (see above). |
| `batch_size` | 512 leaves the GPU idle. 32k+ is the right order for this model size. Scale the learning rate with it, or use `optimizer: adamw`. |
| Validation accumulators | `val_loss` is accumulated as two device scalars and reduced with one two-element collective. Previously every validation feature, label, weight and logit was copied to the CPU every epoch, and under DDP the full feature matrix was `all_gather`-ed. The full arrays are now only materialised on epochs where a diagnostic is actually due. |
| `matmul_precision: high` | TF32 on the tensor cores. |
| `save_last: false` | `last.ckpt` was being rewritten every epoch. |
| `.npy` snapshot | The ensemble dataset snapshot is memory-mapped instead of being parsed into a private copy per worker. |

Measure it on your own data and hardware:

```
python -m nsbi_carl.benchmark --events 2000000 --members 8 --epochs 5
```

It times the old DataLoader path, device batching, and the vectorized
ensemble against each other on synthetic data of the shape you give it.

`tests/smoke_test.py` runs both ensemble modes and the single-network path
end to end on synthetic `.h5` inputs and checks that stacked inference agrees
with the per-model loop.

## Programmatic use

```python
from nsbi_carl import Pipeline

pipe = Pipeline.from_yaml("configs/example.yaml")
dataset, splits = pipe.prepare_data()   # steps 1-4 only
# ... or pipe.run() for everything
```
