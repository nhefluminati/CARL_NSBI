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

### Negative target weights: the neural resampler

For targets with many negative weights, `nsbi_carl.resample` (Nachman &
Thaler, arXiv:2007.11586) writes a positive weight branch into the input
files. It regresses `<w|x>` and `<w²|x>` on the CARL features with k-fold
cross-fitting, then keeps each event with probability `1/K`,
`K = <w²|x>/<w|x>²`, with weight `<w|x>·K` (dropped events get 0). That
preserves the density in the CARL features — hence `s/(1-s) = p_t/p_r` —
and the per-region sum of `w²`. `mode: mean` instead uses `<w|x>` for
every event (lower variance, but no longer a valid MC stat uncertainty).

```bash
python -m nsbi_carl.resample --config configs/qq_4l.yaml   # adds 'weight_positive'
# then in that config:  data.weight_key: weight_positive
```

Options go in an optional `resampler:` block of the same config (see the
module docstring). The command writes closure plots and a diagnostics YAML
to `<output_dir>/resampler/`. Check the closure plots before training on
the new weights. `tests/resample_test.py` covers the method.

`method: cell` uses cell resampling (Andersen & Maier, arXiv:2109.07851)
instead. No networks: around every negative-weight seed it gathers nearest
neighbours until the summed weight is ≥ 0, then sets each member to
`|w_i|·Σw/Σ|w|`. Each cell's total is conserved exactly, but the
distributions are smeared on the cell scale (core deficit, excess in the
tails), and in many features the cells stay wide. `cells_<file>.png` shows
how far they reach per feature. The paired closure errors account for
correlations inside a cell.

If features use fill values for undefined quantities (e.g. `dijet_* = -999`
without two jets), set `resampler.sentinel_below: -900`. Otherwise the
standardisation squashes the physical range and the regression cannot
resolve it. The closure test uses paired per-event errors
(`sqrt(Σ(w_pos − w)²)` per bin), because both histograms come from the same
events. A chi²/ndf well above 1 there means real bias, not noise.

## Model

`CARL(LightningModule)`: configurable `n_layers` / `n_nodes` / `dropout`,
swish (SiLU) hidden activations, single sigmoid output, weighted BCE loss,
SGD optimizer, `CosineAnnealingWarmRestarts(T_0, T_mult, eta_min)` scheduler.
Train/val loss are logged each epoch.

### Loss: `bce` or `pare`

`model.loss: pare` trains with the pole-adjustable ratio estimation loss of
Drnevich, Jiggins, Katzy & Cranmer, *Mach. Learn.: Sci. Technol.* 6 (2025)
045023:

```
L(s, y; t) = (1 - s * t_y)^2          s = raw network output (no sigmoid)
s*         = (t0 + t1 r) / (t0^2 + t1^2 r)
r          = t0 (1 - t0 s) / (t1 (t1 s - 1))
```

Why it helps with negative weights: BCE's sigmoid cannot express `r < 0`, and
wherever the weighted target density is locally negative the BCE has no
minimum (the optimiser drives those events' score to 0). The PARE loss is
quadratic in `s`; its population minimum exists and is the ratio for every
`r > r_pole = -(t0/t1)^2`, negative ratios included. It is the same loss in
both training paths (Lightning and vectorized ensemble).

**Output parametrization (`model.pare.param`).** Training the paper's raw
output from scratch does not work with a large `|t0/t1|`: with `t = (25619,
58)` every ratio in [0, 100] maps into an output window ~1e-5 wide around
`1/t0`, right next to the decode pole at `s = 1/t1`. The optimiser's steps
on a wide network are far coarser than that window, so the output jitters,
warm restarts throw it across the pole, and the decoded ratios explode (the
paper itself only uses PARE to fine-tune pretrained RoSMM models). The
default `param: ratio` therefore has the network output `u` with
`s = (1 + rho u)/t0`, `rho = t1/t0`, and trains on PARE/rho^2 minus its
constant:

```
l(u, y=0) = u^2          l(u, y=1) = rho^2 u^2 - 2 (1 - rho) u
u*        = r (1 - rho) / (1 + rho^2 r)  ~  r          r = u / (1 - rho - rho^2 u)
```

Same minimiser, same pole, but the output is O(r) and the loss O(1). The
loss can be negative; that is expected (it is ~ -E_target[r] at the
optimum). `param: raw` keeps the paper's form, and old PARE checkpoints
load as raw.

```yaml
model:
  loss: pare
  pare: {t0: 25619, t1: 58, param: ratio}   # the paper's t: r_pole ~ -1.95e5
training:
  optimizer: adamw               # param: raw with SGD warns (gradients scale with t)
  min_delta: null                # -> 0 for pare
performance:
  precision: 32-true             # bf16/fp16 is refused for pare
```

Choosing `t`: only `rho = t1/t0` shapes the problem. With `param: ratio` the
estimator is linear in `r` for `|r| << |r_pole| = 1/rho^2`, so the default
leaves ample room; a smaller `|r_pole|` only matters if it approaches the
ratios you need. With `param: raw` follow the paper's App. A.4: keep `|t0/t1|`
well above the largest `|r|`, but not so far that the loss flattens.

Things that behave differently with PARE:

* **Output decoding.** Checkpoints store `loss`, `pare_t0`, `pare_t1`. Every
  consumer (`CARL.forward`, ensemble scoring, `inference.EnsembleScorer`)
  decodes the raw output and re-encodes the ratio as `s = r/(1+r)`, so
  `r = s/(1-s)` holds unchanged downstream. Old checkpoints load as BCE.
  Members with different loss settings cannot be stacked together.
* **Negative ratios downstream.** A negative PARE ratio is stored as a
  score outside [0, 1] (`s < 0` for `-1 < r < 0`, `s > 1` for `r < -1`).
  Convert scores with `combine.ratio_from_scores`, never with a (0, 1) clip:
  clipping turns every `r < -1` into `r ~ 1e12`. The evaluation context,
  the integral metric (now signed) and the diagnostics use it. PARE
  ensembles combine members in ratio space (`mean_score` = `mean_ratio`);
  `mean_logit` is refused. Code of your own that reads scores (likelihood
  scans) must do the same.
* **Overfitting.** PARE rewards memorization far more than BCE: an isolated
  target event the network can carve out lowers the loss by up to
  `w (1-rho)^2/rho^2` (~2e5 w at the default t), against at most `w log 2`
  for BCE. A wide network without regularisation finds those events within
  a few dozen epochs (validation loss climbing at the end of each cosine
  cycle, training loss far below -1). Use `training.weight_decay`; 1e-3
  removed the effect entirely in a 4x256 toy with 7 noise features, while
  a smaller `|t0/t1|` or a bounded output only softened it. The pipeline
  warns when PARE runs with `weight_decay: 0`.
* **Loss values.** Validation sums are accumulated in float64, logs print
  more digits, and the early-stopping `min_delta` defaults to 0 instead of
  1e-6 (with `param: raw` the loss sits near a constant and improves by
  O((t1/t0)^2)).

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
analysis time with `EnsembleScorer(record, combiner="mean_logit")` or
`scorer.set_combiner(...)` — no retraining needed. See `nsbi_carl/combine.py`.
Beware `mean_ratio`: it is unbounded, so a single member whose sigmoid
saturates (s = 1.0 in float32 already at logit ~17) sets the ensemble ratio
for that event on its own; it warns when that happens.

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
