# ARE-005 — Implicit feedback and user-disjoint ML dataset

> Historical v1 report. Superseded by [ARE-005.1](ARE-005.1-evaluation-protocol.md).
> The current default is rating >=5; evaluation splits RAW history before feedback
> and retains all assigned users. The >=7 and pre-holdout eligibility figures below
> describe the original implementation only, NOT the current protocol/configuration.

## Scope and modules

Only prepared local ARE-004 Parquet is used. No source downloads, YummyAnime
requests, ALS fitting, ranking, MLflow, databases or services are involved.

- `preprocessing/implicit_feedback.py`: source validation, rating threshold,
  binary preference and uniform confidence. Raw ratings remain unchanged.
- `preprocessing/data_split.py`: deterministic user cohorts, activity eligibility,
  train-only iterative k-core and per-user context/target assignment.
- `preprocessing/sparse_matrix.py`: sorted train-user/item mappings and CSR.
  The item mapping includes MAL ID but columns are not raw MAL IDs.
- `evaluation/dataset_validation.py`: disjointness, exhaustive holdouts, train-only
  vocabulary, sparse coordinates and cold-item statistics.
- `training/dataset.py`: configuration, local loading, orchestration, artifact
  persistence and verified restoration. Reuses ARE-004 loader/checksum/writers.
- `notebooks/03_ml_dataset_preparation.ipynb`: parameters, module calls and display.
  No algorithm implementations or HTTP requests in cells.

## Run

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml]"
.venv\Scripts\python.exe -m anime_recommender.training.dataset
# Custom configuration, resolved relative to project root:
.venv\Scripts\python.exe -m anime_recommender.training.dataset --config configs/ml_dataset.json
```

Notebook additionally needs the research extra: `.[dev,data,ml,research]`.
It uses the local .venv kernel. `rebuild=True` runs the offline preparation;
False loads existing certified artifacts. Neither setting downloads input data.
The ordinary entry point does not import an ALS implementation or fit any model.

## Feedback contract

Default positive threshold: rating >=7. Thresholds 6 and 8 are supported and
summarized without fitting models. Only observed positive rows are returned;
zero and missing scores are not treated as dislikes. Clean source files are read
only, and source rating/rating_raw columns remain available unchanged.

Added columns: `positive=1`, `confidence=1.0` (float32). Arbitrary rating weights,
negative confidences or another feedback mode are rejected in this version.
The confidence column is the extension point for future validated weighting
experiments; no weighting interface hierarchy is introduced prematurely.
Repeated user-item pairs are an input error, not silently summed in CSR.

## Split protocol and filtering order

1. Use all source user IDs, including users with no qualifying positives, to
   assign 80% train, 10% validation and 10% test BEFORE activity filtering.
   Sort users by SHA-256(seed, domain="users", user_id), with ID tie-breaker.
   Largest-remainder rounding gives exact cohort counts where possible.
   Fixed input IDs and seed give an order-independent reproducible split.
2. Within each cohort require at least 5 positive interactions. Excluded users
   remain recorded in user_assignments; cohorts are NOT rebalanced afterward.
   This user-local eligibility does not decide the item vocabulary.
3. On train alone, iteratively remove users with <5 positives and items with <5
   distinct training users until neither side changes. Every round is reported.
   Validation/test items or frequencies are never inputs to this procedure.
4. Make the item vocabulary from the FINAL train rows only. Stable matrix row
   and column indices are assigned by ascending dataset IDs, starting at zero.
   Join the already validated catalog mapping to attach MAL IDs to columns.
5. For each eligible validation/test user, sort their positives by a separately
   seeded user/item hash. Put floor(0.8*n) into context, clamped to configured
   nonempty context/target minima. Rounding means global ratios need not be 80/20.
   No timestamps or original CSV order are interpreted as interaction chronology.
6. Mark each eval row is_cold_item according to the train vocabulary. Keep ALL
   eligible context and target rows, including cold items. Do not filter users
   again by their number of known items. Report empty-known-context/target users.

Future fold-in must use ONLY context rows with is_cold_item=False; target is
never an input to a user vector. No vectors are computed in ARE-005. Evaluation
targets that are unknown to train remain unavoidable misses in an end-to-end
ALS-only evaluation. Conditional known-item metrics must be labeled separately.

The default training sample is sufficient for the configured 5/5 thresholds.
No automatic relaxation is implemented. Empty train causes a clear failure;
empty eligible eval cohorts are saved with explicit warnings and no reallocation.

## Validation and artifacts

Validation checks unique assignments, all three disjoint user sets, correct
cohort ownership, context/target pair disjointness, nonempty sides for included
users, and complete union of eligible held-out positives. It also checks cold
flags, stable contiguous mappings, exact train-only vocabulary, CSR dimensions,
nnz, all binary confidences and exact sparse row/column coordinates.
No dense user-item matrix is constructed during creation or validation.

`data/processed/ml/` (all ignored by Git):

```text
train_interactions.parquet
validation_context.parquet
validation_target.parquet
test_context.parquet
test_target.parquet
user_assignments.parquet
train_user_mapping.parquet
validation_user_mapping.parquet
test_user_mapping.parquet
item_mapping.parquet
threshold_comparison.parquet
train_matrix.npz
config.json
dataset_report.json
```

Parquet preserves table types and source ratings. NPZ is a compressed SciPy CSR,
not a model. User assignment records include activity and inclusion flags.
Each cohort has its own local user-row indices; all cohorts use the SAME item
vocabulary. `item_mapping` has dataset_anime_id, matrix_col, myanimelist_id.

The report carries input hashes, configuration, package versions, all cohort and
filter counts, cold-item shares, invariants and output SHA-256 values. The report
is written last; failed runs have status=failed and do not certify old outputs.
Writers replace each individual file atomically; the artifact directory is not a
multi-file transaction. Concurrent writers to one output directory are unsupported.
Bad configuration is rejected before trusting its output path.
`load_dataset(path)` verifies hashes and restores all Parquet and CSR artifacts.
Optional expected_config and expected_source_sha256 reject stale artifacts. The
notebook supplies both when rebuild=False, so a changed threshold or changed
ARE-004 input cannot silently be mixed with an older split and matrix.

## Real run on 2026-11-12

Input: 75,900 interactions, 1,000 users. Positive counts BEFORE activity filtering:

| Threshold | Interactions | Users | Anime |
|---:|---:|---:|---:|
| 6 | 66,326 | 996 | 5,738 |
| 7, default | 58,021 | 995 | 5,098 |
| 8 | 44,204 | 983 | 4,388 |

| Cohort | Allocated users | Eligible before train core | Final users | Final positives | Context | Target |
|---|---:|---:|---:|---:|---:|---:|
| Train | 800 | 770 | 761 | 39,540 | n/a | n/a |
| Validation | 100 | 99 | 99 | 6,348 | 5,035 | 1,313 |
| Test | 100 | 98 | 98 | 6,757 | 5,367 | 1,390 |

Thirty train, one validation and two test users fail the initial minimum; their
positive rows number 81, 3 and 6 respectively (90 total). Train k-core removes
another 5,286 rows and 9 users over two changing rounds plus a final stable round.
Final train has 1,718 anime, versus 2,225 anime in validation and 2,222 in test.
No user is moved between cohorts to make the final sizes appear 80/10/10.

Accounting: 58,021 = 39,540 + 6,348 + 6,757 + 90 + 5,286.
All other 17,879 clean rows are below the positive threshold or missing; source
rows remain available in ARE-004 rather than being overwritten or called dislikes.

| Cold-item statistic | Validation | Test |
|---|---:|---:|
| Cold interactions, context+target | 1,029 / 6,348 (16.21%) | 1,002 / 6,757 (14.83%) |
| Cold context rows | 820 / 5,035 (16.29%) | 793 / 5,367 (14.78%) |
| Cold target rows | 209 / 1,313 (15.92%) | 209 / 1,390 (15.04%) |
| Unique cold anime | 895 / 2,225 (40.22%) | 828 / 2,222 (37.26%) |
| Users without known context | 0 | 0 |
| Users without known target | 0 | 0 |

CSR shape: (761,1718), nnz=39,540, dtype=float32, sparsity=96.9757%.
All user-disjoint, context-target, exhaustive-holdout and CSR checks passed.

Two real runs with requests.get blocked produced identical reports and SHA-256
artifacts, including NPZ. All Parquet tables and the matrix were restored and
compared. Notebook 03 ran top-to-bottom in a Jupyter kernel with network forbidden;
its executed copy remained in RAM, and the saved notebook has no outputs.
Existing ARE-004 artifacts remain checksum-verified and unchanged.

Final test run: 82 passed; pip check found no broken requirements. Tests use
synthetic data, including all-cold eval users, empty eval cohorts, injected
context/target leakage, train-only vocabulary under eval-item mutation, corrupted
artifacts and stale configuration/source fingerprints. No test calls a real API.
Saved notebook outputs are empty. No commit/push was performed.

## Limits

- No interaction timestamps: this is preference reconstruction, not prediction
  of future viewing. Random context can chronologically follow a hidden target.
- Hash cohort allocation is reproducible for a fixed input user population;
  adding/removing users can change allocation. Freeze the input hashes for comparisons.
- Threshold changes can change eligibility and vocabulary; report these changes
  rather than comparing scores on silently different evaluation populations.
- This is a 1,000-user development sample. Low-frequency items and active-user
  eligibility bias the usable catalog; coverage is not recommendation quality.
- Only positive holdouts are prepared. Future candidate filtering of other known
  source ratings can use ARE-004 and the saved assignments, without calling them
  negatives or using target to compute the fold-in vector.
- Artifact byte reproducibility is tested within the recorded environment.
  Library upgrades may alter serialization even when logical tables agree.
- No actual ranking metrics, factors, recommendations or model quality are claimed.
