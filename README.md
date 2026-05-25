# Recommender Regime Pipeline

This repository contains the experiment pipeline used to construct dataset regimes, train RecBole models, and evaluate sampled-candidate Top-K recommendation metrics for the master’s thesis:

**Exploring Performance Variability in Recommender Systems: The Role of Sparsity and Dataset Characteristics**

The code is organized around two phases:

1. **Variant preparation** with `prepare_variants.py`.
2. **Model training and external evaluation** with `recbole_models.py`.

The shared configuration and evaluation utilities are in `common.py`. Dataset-specific conversion helpers are provided for LastFM, Online Retail, and Steam.

---

## Repository contents

| Path | Purpose |
|---|---|
| `common.py` | Shared configuration, input normalization, split utilities, candidate sampling, ranking metrics, and artifact paths. |
| `prepare_variants.py` | Builds dataset variants: fixed-users item head/tail filtering, fixed-items user head/tail filtering, k-core variants, random-drop variants, and k-core size-matched variants. |
| `recbole_models.py` | Trains and evaluates BPR, ItemKNN, LightGCN, NeuMF, and MultiVAE through RecBole, followed by external sampled-candidate evaluation. |
| `preprocessing_lastfm.py` | Converts a LastFM user-item-playcount TSV to an implicit interaction CSV. |
| `preprocessing_online_retail.py` | Converts Online Retail transactions to an implicit interaction CSV. |
| `preprocessing_steam.py` | Converts Steam dictionary-style review logs to an implicit interaction CSV. |
| `requirements_minimal.txt` | Minimal package list for the Python environment. |
| `results/main_results.csv` | Cleaned aggregate main-regime result file used for thesis tables. |
| `results/results_kcore_sizematch.csv` | Cleaned aggregate size-matched k-core control result file used for thesis tables. |
| `results/results_aggressive_taildrop.csv` | Cleaned aggregate auxiliary aggressive tail-drop result file used for thesis appendix tables. |

The repository `results/` directory contains the cleaned aggregate CSV files used in the thesis tables. Raw per-run RecBole outputs are not included; the aggregation rules are documented in the thesis and in the result file headers.

The final thesis version does **not** use the earlier KC50 add-back diagnostic as thesis evidence, so add-back result files are intentionally not part of the final result set.

---

## Version identity, Git commit, and SHA256 hashes

For thesis submission and later auditability, the repository state should be identified by a fixed Git commit or release tag, not only by the mutable `main` branch.

Recommended commands after the final repository update:

```bash
git rev-parse HEAD
git tag v1.0-thesis-submission
git push origin v1.0-thesis-submission
```

If SHA256 prefixes are reported in the thesis, recompute them after the final files are fixed:

```bash
sha256sum common.py prepare_variants.py recbole_models.py \
  preprocessing_lastfm.py preprocessing_online_retail.py preprocessing_steam.py \
  results/*.csv requirements_minimal.txt > artifact_manifest_sha256.txt
```

The PDF can report short SHA256 prefixes for compactness, but the complete hashes should be stored in `artifact_manifest_sha256.txt`. If the thesis table and the GitHub files differ, treat the thesis SHA256 prefixes as artifact-specific identifiers and regenerate the table from the final submitted artifact.

---

## Expected interaction format

The main pipeline expects a CSV that can be normalized to the following columns:

```text
user_id,item_id,rating[,timestamp]
```

`timestamp` is optional. If it is available, the leave-one-out split can be chronological. If it is missing, the split falls back to random leave-one-out under the configured seed.

Explicit-rating datasets can be binarized into implicit positives. The default rule is:

```text
rating >= ceil(0.8 * RATING_MAX)
```

For a 1–5 rating scale, this keeps ratings 4 and 5.

---

## Output layout

All outputs are written under:

```text
$EXPORT_BASE/$DATASET_NAME/$RUN_ID/
```

The main subdirectories are:

```text
variants/       materialized ratings_<variant>.csv files
splits/         train.csv, test.csv, and meta.json for each variant
artifacts/      RecBole input files, model states, metrics, and optional recommendation lists
results/        recbole_results_by_seed*.csv and recbole_results_summary*.csv
meta/           run-level metadata
```

Use the same `RUN_ID` for preparation and model training. If the two phases are started from different terminal sessions, export the same `RUN_ID` manually.

---

## Installation

Use the same Python environment for all scripts. A minimal environment needs:

```bash
pip install numpy pandas joblib scipy torch recbole scikit-surprise
```

`pyarrow` is optional and only used when parquet export is available.

---

## Core environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `INPUT_RATINGS_PATH` | `~/datasets/movielens/rating.csv` | Input interaction CSV. |
| `EXPORT_BASE` | `~/exports_sparsity_new` | Base directory for all outputs. |
| `DATASET_NAME` | inferred from input path | Dataset label used in output paths. |
| `RUN_ID` | timestamp | Run identifier. Set this explicitly for reproducibility. |
| `SEED` | `42` | Main random seed. |
| `SEEDS` | value of `SEED` | Comma-separated seeds for model training, e.g. `42,43,44`. |
| `TOPK` | `10` | Recommendation cutoff. |
| `IMPLICIT_MIN_POS_PER_USER` | `0` | Optional minimum positives per user after binarization. |
| `BINARIZE_FROM_EXPLICIT` | true for MovieLens-like names | Enables explicit-to-implicit thresholding. |
| `BINARY_THRESHOLD_FRACTION` | `0.8` | Fraction of `RATING_MAX` used for the positivity threshold. |
| `RATING_MIN`, `RATING_MAX` | `1`, `5` | Rating scale used by explicit datasets. |
| `NEGATIVE_CANDIDATE_SAMPLE` | `1000` | Number of sampled negative candidates per evaluated user. Use `0` for full-catalog ranking when feasible. |

---

## Variant preparation switches

The preparation phase is controlled by:

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
```

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_PREPARE_VARIANTS` | `1` | Enables variant construction when running `prepare_variants.py`. |
| `PREP_ITEM_VARIANTS` | `1` | Builds fixed-users item-side variants: `raw_aligned`, `sparse`, `dense`. |
| `PREP_USER_VARIANTS` | `1` | Builds fixed-items user-side variants: `raw_aligned_fixitems`, `user_sparse`, `user_dense`. |
| `SAVE_RAW_FULL` | `0` | Saves the normalized full input as `ratings_raw_full.csv`. |
| `TARGET_USERS` | `20000` | User target used by aligned or fixed-support regimes when feasible. |
| `TARGET_ITEMS` | `5000` | Item target used by fixed-items and random-drop support construction when feasible. |
| `ITEM_HEAD_DROP_PCT` | `0.10` | Fraction of most popular items removed in fixed-users head-drop. |
| `ITEM_TAIL_DROP_PCT` | `0.40` | Fraction of least popular items removed in fixed-users tail-drop. |
| `USER_HEAD_DROP_PCT` | `0.10` | Fraction of most active users removed in fixed-items head-user drop. |
| `USER_TAIL_DROP_PCT` | `0.40` | Fraction of least active users removed in fixed-items tail-user drop. |
| `PREP_KCORE_VARIANTS` | `0` | Builds k-core variants. |
| `KCORE_KS` | `5,10,50` | k values for k-core construction. |
| `KCORE_SAVE_PLAIN` | `0` | Saves unaligned `ratings_kcore<k>_plain.csv` files. Use this for the main k-core regime. |
| `KCORE_SAVE_RAW_BASELINE` | `0` | Saves `ratings_raw_kcore_aligned.csv` when aligned k-core exports are used. |
| `PREP_RANDOM_DROP_VARIANTS` | `0` | Builds random-drop fixed-support variants. |
| `RANDDROP_BASE` | `raw` | Base graph for random drop. Use `raw` or `kcore<k>`, for example `kcore10`. |
| `RANDDROP_PCTS` | `0.2,0.4,0.6` | Edge-drop fractions. |
| `RANDDROP_MIN_USER` | `3` | Minimum retained user degree during random drop. |
| `RANDDROP_MIN_ITEM` | `2` | Minimum retained item degree during random drop. |
| `PREP_COMMON_SPLITS` | `1` | Creates train/test splits after variant construction. |
| `COMMON_SPLIT_MODE` | `auto` | `auto`, `chrono_loo`, or `random_loo`. |
| `COMMON_SPLIT_TIME_COL` | `timestamp` | Timestamp column used by chronological leave-one-out. |
| `COMMON_SPLIT_SEED` | `SEED` | Split seed. |
| `COMMON_SPLIT_SHUFFLE` | `0` | Shuffles interactions before random leave-one-out. |
| `FORCE_COMMON_SPLIT_OVERWRITE` | `0` | Rewrites existing split files. |
| `COMMON_SPLIT_FILE_GLOB` | empty | Optional comma-separated glob filter for split creation. |
| `RUN_SPARSITY_REPORTS` | `1` | Prints structural reports after variant construction. |
| `SPARSITY_REPORT_FILE_GLOB` | empty | Optional comma-separated glob filter for sparsity reports. |
| `FORCE_EXPORT_OVERWRITE` | `0` | Rewrites existing variant CSVs. |

---

## K-core size-matched control

The code includes a k-core size-matching preparation path in `prepare_variants.py` through `prep_kcore_size_match`. This is the preparation step used for the thesis size-matched control: KC5 and KC10 are sampled to the same user and item cardinality as KC50.

Relevant switches:

| Variable | Default | Meaning |
|---|---:|---|
| `PREP_KCORE_SIZE_MATCHED_VARIANTS` | `0` | Enables size-matched k-core variant preparation. |
| `PREP_ONLY_KCORE_SIZE_MATCH` | `0` | Disables the other variant families and runs only size-matched k-core preparation. |
| `KCORE_SIZE_MATCH_TARGET` | `50` | Target k-core whose user/item cardinality is matched. |
| `KCORE_SIZE_MATCH_BASE_KS` | `5,10` | Source k-core levels to be sampled down to the target size. |
| `KCORE_SIZE_MATCH_SEEDS` | `42,43,44` | Sampling seeds for stochastic size-matched variants. |
| `KCORE_SIZE_MATCH_INCLUDE_TARGET` | `1` | Includes the target KC50 plain reference in the exported size-match set. |
| `KCORE_SIZE_MATCH_MAX_ATTEMPTS` | `200` | Maximum sampling attempts for exact user/item cardinality matching. |

Example:

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0

export PREP_ONLY_KCORE_SIZE_MATCH=1
export PREP_KCORE_SIZE_MATCHED_VARIANTS=1
export KCORE_SIZE_MATCH_TARGET=50
export KCORE_SIZE_MATCH_BASE_KS="5,10"
export KCORE_SIZE_MATCH_SEEDS="42,43,44"
export KCORE_SIZE_MATCH_INCLUDE_TARGET=1

export PREP_COMMON_SPLITS=1
export FORCE_COMMON_SPLIT_OVERWRITE=1

python -u prepare_variants.py 2>&1 | tee "prepare_sizematch_${RUN_ID}.log"
```

Expected variant names include:

```text
ratings_kcore5_sizematch_kcore50_seed42.csv
ratings_kcore5_sizematch_kcore50_seed43.csv
ratings_kcore5_sizematch_kcore50_seed44.csv
ratings_kcore10_sizematch_kcore50_seed42.csv
ratings_kcore10_sizematch_kcore50_seed43.csv
ratings_kcore10_sizematch_kcore50_seed44.csv
ratings_kcore50_plain.csv
```

During model execution, select these variants with `RUN_VARIANT_GLOB` if needed.

---

## RecBole model switches

The model phase is controlled by:

```bash
export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
```

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_RECBOLE_MODELS` | `1` | Enables RecBole training/evaluation. |
| `RECBOLE_MODELS` | all five models | Optional comma-separated subset read by `recbole_models.py`, e.g. `BPR,LightGCN`. |
| `RUN_ITEM_VARIANTS` | `1` | Includes `raw_aligned`, `sparse`, and `dense`. |
| `RUN_USER_VARIANTS` | `1` | Includes `raw_aligned_fixitems`, `user_sparse`, and `user_dense`. |
| `RUN_RAW_FULL_VARIANT` | `0` | Includes `raw_full` if present. |
| `RUN_KCORE_VARIANTS` | `1` | Includes k-core variants. |
| `RUN_KCORE_PLAIN_ONLY` | `1` | Uses only `ratings_kcore*_plain.csv` files when running k-core variants. |
| `RUN_RANDDROP_VARIANTS` | `0` | Includes `ratings_randdrop*.csv`. |
| `RUN_VARIANT_GLOB` | empty | Optional comma-separated filter over variant file names or variant names. |
| `RUN_TAG` | empty | Suffix used in output result CSV names. |
| `RECBOLE_DEVICE` | `cpu` | Device passed to RecBole, for example `cuda:0`. |
| `RECBOLE_EPOCHS` | `20` | Number of training epochs. |
| `RECBOLE_TRAIN_BATCH_SIZE` | `1024` | Default train batch size. |
| `RECBOLE_EVAL_BATCH_SIZE` | `2048` | Default evaluation batch size. |
| `RECBOLE_TRAIN_BATCH_SIZE_MULTIVAE` | `512` | MultiVAE train batch size. |
| `RECBOLE_EVAL_BATCH_SIZE_MULTIVAE` | `1024` | MultiVAE evaluation batch size. |
| `RB_EVAL_BATCH_SIZE` | `2048` | Batch size used by the external RecBole scoring adapter. |
| `RECBOLE_EVAL_STEP` | `5` | RecBole internal evaluation step. Ignored when validation is skipped. |
| `RECBOLE_EVAL_SAMPLE_NUM` | `200` | RecBole internal negative sample count. Reported metrics come from external evaluation. |
| `SKIP_FIT_VALIDATION` | `1` | Skips internal validation by setting an effectively unreachable evaluation step. |
| `REUSE_ARTIFACTS` | `1` | Reuses existing per-model metrics if present. |
| `EVAL_ONLY` | `0` | Recomputes metrics from saved Top-K files without retraining. Requires `SAVE_RECS=1` from a previous run. |
| `SAVE_RECS` | `1` | Saves Top-K recommendations under each model artifact directory. |
| `SAVE_TOPK_JSON` | `0` | Saves additional Top-K JSON output from the shared evaluator. |
| `OVERWRITE_METRICS` | `1` | Overwrites metric files in eval-only mode. |
| `COMPUTE_BEYOND_ACCURACY` | `0` | Enables additional beyond-accuracy metrics. The final main thesis runs used `1`. |
| `COMPUTE_ILS` | `0` | Enables ILS/Diversity computation. The final main thesis runs used `1`; this can be expensive. |
| `LOG_SPARSITY_PER_MODEL` | `0` | Logs extra sparsity reports per model. |

The defaults are conservative. To reproduce the thesis main metrics, explicitly enable the beyond-accuracy diagnostics used by the submitted runs:

```bash
export COMPUTE_BEYOND_ACCURACY=1
export COMPUTE_ILS=1
```

Adapter debug note: the submitted `recbole_models.py` snapshot may print a small number of `[DBG]` lines from the external RecBole scoring adapter. This is expected diagnostic output in the submitted code snapshot, not a documented environment-variable switch, and it does not affect the computed metrics.

---

## Standard workflow

### 1. Set a run ID

```bash
export RUN_ID=my_experiment_$(date +%Y%m%d_%H%M%S)
```

Keep this value for both phases.

### 2. Build variants and splits

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"
```

### 3. Train and evaluate models

```bash
export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

---

## Example: MovieLens main regime run

```bash
export EXPORT_BASE=~/projects/mrs_movielens/exports
export DATASET_NAME=movielens
export INPUT_RATINGS_PATH=/home/majnar-csoban/student1/exports/ratings_ml20m.csv

export RUN_ID=ml20m_main_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=ml20m_main

export BINARIZE_FROM_EXPLICIT=1
export BINARY_THRESHOLD_FRACTION=0.8
export RATING_MIN=1
export RATING_MAX=5

export SEED=42
export SEEDS="42,43,44"
export TOPK=10
export NEGATIVE_CANDIDATE_SAMPLE=1000

export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0

export SAVE_RAW_FULL=1
export PREP_ITEM_VARIANTS=1
export PREP_USER_VARIANTS=1
export PREP_KCORE_VARIANTS=1
export KCORE_KS="5,10,50"
export KCORE_SAVE_PLAIN=1
export PREP_RANDOM_DROP_VARIANTS=1
export RANDDROP_PCTS="0.2,0.4,0.6"

export PREP_COMMON_SPLITS=1
export COMMON_SPLIT_MODE=auto
export COMMON_SPLIT_TIME_COL=timestamp
export COMMON_SPLIT_SEED=42
export FORCE_COMMON_SPLIT_OVERWRITE=1

export RUN_SPARSITY_REPORTS=1

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"

export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1

export CUDA_VISIBLE_DEVICES=0
export RECBOLE_DEVICE=cuda:0

export RECBOLE_MODELS="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"
export RUN_ITEM_VARIANTS=1
export RUN_USER_VARIANTS=1
export RUN_KCORE_VARIANTS=1
export RUN_KCORE_PLAIN_ONLY=1
export RUN_RANDDROP_VARIANTS=1

export REUSE_ARTIFACTS=0
export SKIP_FIT_VALIDATION=1
export RECBOLE_EPOCHS=20
export SAVE_RECS=1
export OVERWRITE_METRICS=1

export COMPUTE_BEYOND_ACCURACY=1
export COMPUTE_ILS=1

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

---

## Dataset preprocessing examples

### LastFM

```bash
python preprocessing_lastfm.py --in inter.txt --out ratings.csv
```

### Online Retail

```bash
python preprocessing_online_retail.py --in OnlineRetail.csv --out ratings_raw_full.csv
```

Use `--session-as-user` only if each invoice should be treated as a separate user.

### Steam

Inspect the first records:

```bash
python preprocessing_steam.py --in steam_reviews.json --out rating.csv --inspect
```

Convert the full file:

```bash
python preprocessing_steam.py --in steam_reviews.json --out rating.csv --db steam_reviews.sqlite --batch-size 5000
```

---

## Interpreting generated result files

The model script writes two main CSV files inside each run directory:

```text
results/recbole_results_by_seed_<RUN_TAG>.csv
results/recbole_results_summary_<RUN_TAG>.csv
```

The by-seed file contains one row per `variant × model × seed`. The summary file groups by `variant × model` and stores mean and sample standard deviation for numeric metrics.

The primary accuracy metric used by the thesis pipeline is `NDCG@10`. `Precision@10`, `Recall@10`, and `MRR@10` are also produced. Beyond-accuracy metrics are only computed when the corresponding switches are enabled.

The final thesis aggregate CSV files in this repository are cleaned result tables derived from the submitted runs. They are intended for audit and table reproduction, not as raw RecBole run directories.

---

## Aggregation conventions used in the thesis tables

The main thesis result tables use model-level seed aggregation unless a table explicitly states otherwise.

For a `dataset × variant × model` group:

1. Average metric values over available seeds.
2. Report sample standard deviation with `ddof=1` when at least two seeds are available.
3. Leave standard deviation blank or unavailable when only one seed exists.
4. Do not average over models unless the table explicitly reports a model-averaged overview.
5. Structural fields such as users, items, interactions, density, sparsity, average user degree, and average item degree are taken from the materialized variant or the prepare log.

The cleaned aggregate CSV files in `results/` follow these thesis-level conventions.

---

## Practical notes

- Keep `NEGATIVE_CANDIDATE_SAMPLE` fixed across experiments that you want to compare directly.
- Keep `RUN_ID` fixed between preparation and model execution.
- Use `RUN_VARIANT_GLOB` to run only selected variants without changing the variant-building code.
- Use `RECBOLE_MODELS` for quick sanity checks, for example `RECBOLE_MODELS=BPR`.
- Use `REUSE_ARTIFACTS=0` if you want to force retraining and overwrite cached metrics.
- Use `SAVE_RECS=1` before using `EVAL_ONLY=1`.
- Do not interpret sampled-candidate metrics as full-catalog ranking metrics.
- The original regimes are split after variant construction; they compare regime-specific tasks, not one identical shared test universe.
- The public repository does not include raw source datasets, full model checkpoints, or all intermediate Top-K recommendation JSON files.
