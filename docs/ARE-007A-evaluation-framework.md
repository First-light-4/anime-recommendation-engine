# ARE-007A: recommendation evaluation framework

Implemented locally; awaiting code review. No commit/push, ARE-007B, tuning, production model training, source downloads, external services or YummyAnime API calls.

## Files and responsibilities

```text
configs/evaluation.json
src/anime_recommender/evaluation/
  ranking_metrics.py       pure binary ranking metrics
  ranking_evaluator.py     protocol guards, fixed-model generation and orchestration
  coverage.py              catalog exposure and context/target availability
  cohorts.py               context-only assignment and history distribution
  reports.py               strict JSON / Parquet save-load with checksums
  dashboard.py             presentation tables, no inference/metric implementations
  recommendation_checks.py legacy diagnostics reuse the shared pure metric kernel
notebooks/06_model_evaluation.ipynb
tests/test_ranking_evaluation.py
docs/ARE-007A-evaluation-framework.md
README.md
```

Existing inference, fold-in, seen filtering, mapping checks and the selective train/validation reader are reused. No production module imports Jupyter. Notebook 01's existing local edit and Notebook 05 were not modified.

## Interfaces and reproduction

```powershell
.venv\Scripts\python.exe -m anime_recommender.evaluation.ranking_evaluator
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

```python
from anime_recommender.evaluation.ranking_evaluator import run_evaluation, evaluate_models
from anime_recommender.evaluation.reports import load_report
from anime_recommender.evaluation.dashboard import dashboard_tables

report = run_evaluation(project_root, config_path="configs/evaluation.json")
loaded = load_report(project_root / report.metadata["report_dir"])
view = dashboard_tables(loaded, k=10)
# Core in-memory API: evaluate_models(bundle, inputs, ks=[5,10,15,20], include_als_fallback=True)
```

run_evaluation loads saved models only. K defaults to [5,10,15,20], is configurable and validated; the split is restricted to validation. Report directories are unique UTC-timestamp/fingerprint paths under ignored artifacts/evaluation/are007a/. Notebook 06 displays tables, curves, cohorts, cold statistics and titled successful/unsuccessful ALS examples. It contains calls, displays and plots rather than business logic.

## Protocol ARE-007A-v1

Generate once at max(K) using train vocabulary minus ALL observed context, including low/zero ratings. Every smaller K uses that list's prefix. Both Full and Warm score exactly the same prefix. Duplicate recommendation IDs and unknown/seen items are rejected. Context-target overlap, unknown evaluation users, duplicate input pairs, model/data mismatch and train-validation user overlap are rejected. Input context must be a subset of observed context.

ALS uses only warm positive context with fixed item factors. The evaluator checks both item/user factors are exactly unchanged. Target is not passed into generation, ranking or fold-in. Popularity uses the existing saved counts of distinct positive train users. Fallback uses already generated Popularity lists for users where ALS is unavailable; it does not refit or invoke a different candidate filter.

The three separately labeled systems are popularity, als (ALS-only) and als_fallback (ALS + explicit Popularity fallback). In this actual validation set all users have warm context, so the two ALS systems coincide. This is not evidence about the benefit of fallback; synthetic tests cover unavailable ALS and fallback differences.

## Metric definitions

Binary relevance is the positive target set R. Precision=hits/K, including when fewer than K candidates exist. Recall=hits/|R|. DCG sums 1/log2(rank+1) at hit ranks; IDCG has min(K,|R|) leading hits. Per-user AP sums hit-rank precisions and divides by min(K,|R|), including when there are more targets than K. MAP is the macro mean of AP. HitRate is 1 if at least one hit, otherwise 0.

All five metrics are undefined when R is empty, including Precision and HitRate. Macro means exclude undefined values and publish eligible counts for each metric. Empty lists with nonempty targets yield zero. K can exceed the available catalog; Precision still divides by K. Cold targets remain in Full Recall/AP/IDCG denominators, so they count as unreachable positives rather than silently disappearing. Warm uses R intersected with train vocabulary and excludes only users without warm targets. It never changes the recommendation list.

Difference from legacy ARE-006 Warm Recall: the new protocol does NOT additionally require warm context for warm-metric eligibility. Thus no-context users with warm targets count as zero for ALS-only and can benefit in explicitly labeled fallback. Legacy diagnostic APIs retain their old condition for compatibility, while sharing the same ranking metric kernel. This eligibility distinction has no numerical effect on the current actual validation snapshot.

Catalog coverage = distinct items exposed at K / 2091 train items. Mean recommended popularity weights displayed recommendation occurrences across users. Warm context coverage = warm positive context count / all positive context count. Warm/cold target shares use the full positive target set, even in Warm diagnostic rows. Zero denominators produce undefined, not fabricated zeros.

Cohorts use only distinct warm positive context lengths: 0, 1, 2-4, 5-9, 10-19 and 20+. Empty groups have N/A metrics and zero user/eligible counts. Targets only contribute post-generation metrics and cold-target summaries, never cohort assignment. Target-free users remain visible.

## Actual validation results

100 users, 2091 train items, rating threshold>=5 and train minimums 5/5 unchanged.

### Full metrics at K=10

| model_name | precision | recall | ndcg | map | hit_rate | recall_eligible_users |
| --- | --- | --- | --- | --- | --- | --- |
| als | 0.230000 | 0.283710 | 0.388166 | 0.274034 | 0.810000 | 100 |
| als_fallback | 0.230000 | 0.283710 | 0.388166 | 0.274034 | 0.810000 | 100 |
| popularity | 0.139000 | 0.212262 | 0.219222 | 0.134116 | 0.650000 | 100 |

### Full/Warm Recall by K

| model_name | mode | k | recall | ndcg | recall_eligible_users |
| --- | --- | --- | --- | --- | --- |
| als | full | 5 | 0.226424 | 0.418757 | 100 |
| als | full | 10 | 0.283710 | 0.388166 | 100 |
| als | full | 15 | 0.346976 | 0.394521 | 100 |
| als | full | 20 | 0.411255 | 0.401817 | 100 |
| als | warm | 5 | 0.231957 | 0.419269 | 100 |
| als | warm | 10 | 0.291948 | 0.389870 | 100 |
| als | warm | 15 | 0.357796 | 0.397655 | 100 |
| als | warm | 20 | 0.424745 | 0.406223 | 100 |
| popularity | full | 5 | 0.138652 | 0.210018 | 100 |
| popularity | full | 10 | 0.212262 | 0.219222 | 100 |
| popularity | full | 15 | 0.244444 | 0.220893 | 100 |
| popularity | full | 20 | 0.279758 | 0.228413 | 100 |
| popularity | warm | 5 | 0.139975 | 0.210018 | 100 |
| popularity | warm | 10 | 0.215081 | 0.219369 | 100 |
| popularity | warm | 15 | 0.248171 | 0.221443 | 100 |
| popularity | warm | 20 | 0.285190 | 0.229641 | 100 |

### Catalog exposure

| model_name | k | recommended_items | catalog_items | catalog_coverage | mean_train_popularity |
| --- | --- | --- | --- | --- | --- |
| als | 5 | 152 | 2091 | 0.072692 | 179.486000 |
| als | 10 | 233 | 2091 | 0.111430 | 160.564000 |
| als | 15 | 298 | 2091 | 0.142516 | 146.697333 |
| als | 20 | 361 | 2091 | 0.172645 | 137.293000 |
| popularity | 5 | 25 | 2091 | 0.011956 | 315.900000 |
| popularity | 10 | 39 | 2091 | 0.018651 | 278.158000 |
| popularity | 15 | 49 | 2091 | 0.023434 | 251.800000 |
| popularity | 20 | 64 | 2091 | 0.030607 | 231.493500 |

### Cohorts at K=10 (Full)

| model_name | cohort | users | recall_eligible_users | recall | ndcg | hit_rate | warm_context_coverage | cold_target_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| als | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| als | 2–4 | 13 | 13 | 0.538462 | 0.469648 | 0.615385 | 1.000000 | 0.000000 |
| als | 5–9 | 14 | 14 | 0.321429 | 0.259476 | 0.571429 | 0.989583 | 0.000000 |
| als | 10–19 | 19 | 19 | 0.373684 | 0.350572 | 0.842105 | 0.971223 | 0.026316 |
| als | 20+ | 54 | 54 | 0.180944 | 0.415142 | 0.907407 | 0.846389 | 0.156516 |
| popularity | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| popularity | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| popularity | 2–4 | 13 | 13 | 0.500000 | 0.299615 | 0.538462 | 1.000000 | 0.000000 |
| popularity | 5–9 | 14 | 14 | 0.273810 | 0.193877 | 0.500000 | 0.989583 | 0.000000 |
| popularity | 10–19 | 19 | 19 | 0.252632 | 0.198170 | 0.578947 | 0.971223 | 0.026316 |
| popularity | 20+ | 54 | 54 | 0.112832 | 0.213845 | 0.740741 | 0.846389 | 0.156516 |

Positive context: 5990 total, 5126 warm (85.576%), 864 cold (14.424%). Positive targets: 1538 total, 1315 warm (85.501%), 223 cold (14.499%). All 100 users have warm context and positive targets; ALS fold-in availability is 100%. Warm target eligibility is also 100 users. No cold item receives fictitious factors.

At K=10 Popularity exposes 39 train items (1.8651%), ALS 233 (11.1430%). Mean displayed train popularity is 278.158 for Popularity and 160.564 for ALS. This describes exposure, not a complex novelty/diversity claim.

### ARE-006 compatibility and reproducibility

Full Recall@10: Popularity 0.21226249830056404; ALS 0.28371001914830907. Full NDCG@10: Popularity 0.21922157608806309; ALS 0.3881661538876461. All four match the supplied six-decimal references within 0.0000005. Warm Recall@10 also matches: Popularity 0.215081, ALS 0.291948.

Every actual result table, recommendation list and example was identical on a repeated run with unchanged inputs and saved models. Timestamps and output directories intentionally differ. No real model was fitted. User/item factors remained unchanged; saved model files and Notebook 01/05 hashes were unchanged across Notebook execution.

## Report formats and privacy

Strict JSON report plus ten Parquet tables: metrics, cohorts, user_metrics, users, history_distribution, coverage, recommendations, statuses, examples and example_users. JSON contains metadata, aggregate metrics/cohorts/coverage/history distribution, nulls for undefined values and SHA-256 hashes for every table. Loading verifies checksums and JSON/Parquet consistency. No pickle. Model configuration, model-file hashes, train fingerprint, exact validation-file hashes, evaluation-data fingerprint, dataset configuration, package versions, protocol definitions, split, K and UTC execution timestamp are recorded.

Validation dataset IDs and per-user recommendations exist only in ignored local Parquet artifacts. No YummyAnime data is collected or persisted. The selective reader accesses train artifacts and four validation artifacts; test targets/context/mappings/matrices are never read or hashed. It reads the existing shared dataset report but does not use test statistics for evaluation or model selection. Integration tests corrupt all synthetic test artifacts and still run evaluation successfully. Configured split=test is explicitly rejected.

## Verification and limitations

136 tests passed in 8.57 seconds, including independent hand-calculated metrics, empty targets/lists, short lists/K above catalog size, duplicates, cold targets, Full/Warm denominators, cohort boundaries, unavailable ALS/fallback, target mutation independence, common seen filters, persistence/tamper checks, deterministic repeats and corrupt/unreadable test artifacts. All existing tests pass after the legacy diagnostic metric refactor.

Notebook 06 executed all cells/plots with HTTP and ALS fit patched to fail. Outputs were kept only in memory. An initial dashboard bracket typo was corrected before the final successful tests/run. Windows Jupyter emitted selector/TCP warnings and a native ZMQ shutdown message; the final notebook run completed with exit code zero.

This is a small sample with random context/target splits and no timestamps: it measures preference reconstruction, not future chronological quality. No confidence intervals, statistical significance claims, temporal validation, tuning or dataset enlargement. Empty real cohorts 0/1 cannot establish real-world fallback performance. Cohort-level target shares are descriptive after assignment. No ARE-007B work or commit/push.

## Final code review and target-size diagnostics

The mathematical audit found no changes required to the metric definitions or
values. Precision uses K, Recall uses the full relevant count, IDCG includes cold
targets in Full, and AP uses min(K, relevant count). MAP is the unweighted macro
mean of per-user AP, not a micro average of hits. All five metrics are undefined
with no relevant targets; eligible counts and null handling remain explicit.
Cold-only users have zero Full metrics for empty hits and undefined Warm metrics.

New regression tests distinguish macro from micro averaging, compare exact
recommendation-call arguments between the models, verify low/zero observed items
are seen-filtered, and verify targets are never generation arguments. The same
maximum-K ordered prediction list is still used for every prefix and both modes.

Added positive_targets_mean/median and warm_targets_mean/median to aggregate and
cohort reports. These descriptive values include ALL users in the group, including
zero-target users; they do not change cohort assignment or metric eligibility.
Empty groups show N/A, not zero target sizes. Notebook 06 adds the target-size table
and a mean/median bar plot, using prepared src tables only.

| Warm context | Users / Full eligible | Full targets mean | Full targets median | Warm targets mean | Warm targets median | ALS Full Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 / 0 | N/A | N/A | N/A | N/A | N/A |
| 1 | 0 / 0 | N/A | N/A | N/A | N/A | N/A |
| 2–4 | 13 / 13 | 1.307692 | 1 | 1.307692 | 1 | 0.538462 |
| 5–9 | 14 / 14 | 2.357143 | 2 | 2.357143 | 2 | 0.321429 |
| 10–19 | 19 / 19 | 4.000000 | 4 | 3.894737 | 4 | 0.373684 |
| 20+ | 54 / 54 | 26.148148 | 17 | 22.055556 | 15.5 | 0.180944 |

The target-size disparity is material: with a fixed K, a longer-history user can
have many more relevant targets, making high Recall harder even when hits improve.
The cohort Recall difference alone does not establish poorer ranking for users
with long histories. Cold targets also contribute to their Full denominator.

The reports already recorded hashes of saved model files; the audit added an
explicit model content fingerprint to cover in-memory bundles as well. It hashes
mapping identity, train fingerprint, ALS configuration, popularity counts, and
user/item factors (including shape/dtype). It is independent of evaluation targets.
Dataset fingerprint: ae675b9a301ab0a9a7437f72ec2071e95a1cb484764870b902b12f2f04edeaab.
Model fingerprint: 4da635752258860aec1efdb39fa456946e17de6116a3e2781642826cb680e781.
Protocol remains ARE-007A-v1: metric definitions are unchanged; report columns and
provenance metadata are extended.

Final verification: 140 tests passed in 5.63 seconds. Repeated actual runs produced
identical tables and dataset/model fingerprints; only timestamps/output paths differ.
ARE-006 rounded Recall/NDCG references still match. JSON/Parquet roundtrips and
integrity checks pass. Saved outputs contain only local public-dataset validation
IDs and diagnostics, not YummyAnime histories, tokens, nicknames or response caches.

Notebook 06 completed every cell and plot with HTTP, fit and Path.open access to
test Parquet/NPZ artifacts forbidden. The source reader still names train and
validation files explicitly. Hashes of all consumed model/train/validation/catalog
artifacts and Notebook 01/05 were unchanged. Outputs were not saved. No commit/push,
model refit, split/threshold change or ARE-007B work was performed.
