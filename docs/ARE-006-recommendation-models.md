# ARE-006: Popularity baseline and CPU ALS

Implemented and verified locally; awaiting code review. No commit or push. No ARE-007 tuning, MLflow, API endpoints, production YummyAnime service, source downloads or test evaluation.

## Files and responsibilities

```text
configs/recommendation_models.json
src/anime_recommender/
  training/
    inputs.py                 selective train/validation reader and fingerprints
    popularity.py             unique positive train-user counts
    als.py                    CPU ALS fit and factor validation
    artifacts.py              model bundle save/load and mapping guards
    pipeline.py               training, reload comparison and diagnostics orchestration
  inference/
    recommendations.py        common output, deterministic ranking, seen exclusion
    popularity.py             baseline inference including empty histories
    fold_in.py                native user-factor solve with fixed items
    als.py                    new-user scoring and explicit unavailability
  evaluation/
    recommendation_checks.py  validation checks and diagnostic metrics
notebooks/04_recommendation_models.ipynb
tests/test_recommendation_models.py
```

README.md and pyproject.toml are updated. Existing ARE-004/005.1 modules are unchanged.
The selective reader avoids the existing whole-dataset loader because it also loads test.
The shared recommendation utility keeps sorting/schema/seen filtering identical.

## Reproduce

Install `.[dev,data,ml,models,research]` in the local virtual environment, then:

```powershell
.venv\Scripts\python.exe -m anime_recommender.training.pipeline
.venv\Scripts\python.exe -m anime_recommender.training.pipeline --load-only
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Notebook 04 defaults to load-only; set retrain=True explicitly for first training, then reset it to False.
It imports project functions and displays their results; no training/fold-in/metric implementation is in cells.
Save the notebook with outputs cleared. Validation recommendations contain dataset user IDs and are stored only in ignored artifacts/.

```python
from anime_recommender.training.artifacts import load_models
bundle = load_models("artifacts/models/are006")
models = bundle.recommenders()
context = bundle.items.dataset_anime_id.head(2).tolist()
result = models["als"].recommend(context, top_k=10, seen_items=context)
# Caller decides whether to use models["popularity"] when result.available is False.
```

IDs in user_context are dataset_anime_id, not MAL IDs or YummyAnime IDs. Future online mapping must explicitly convert via MAL ID. Both outputs contain dataset_anime_id, myanimelist_id, score, rank and model_name.

## Training protocol

Train CSR: users x items = (772, 2091), float32, 48,080 nonzeros. Positive means original rating>=5; train filtering remains 5/5. No full interaction matrix is made dense. Factors are dense by design, and scores are a single item-length vector.

Popularity counts distinct positive train users per item. Ties are broken by dataset_anime_id ascending. No validation information contributes to either model's training.

ALS initial parameters: factors=64, regularization=0.1, iterations=20, random_state=42,
num_threads=1, BLAS threads=1, dtype=float32, alpha=1, use_cg=True, cg_steps=3.
Binary observed confidence is 1. This is an initial baseline policy, not a tuned weighting scheme.
Installed implicit 0.7.3 accepts users x items directly; no transpose. Its recalculate_user solves the regularized least-squares system using fixed item factors, without storing a new train user. For binary context, this is (Y.T Y + lambda I) x = sum(Y_warm). Tests compare against that solve and reject a mean-vector substitute.
See [official CPU ALS documentation](https://benfred.github.io/implicit/api/models/cpu/als.html).

Empty/cold-only context returns available=False, factor=None and reason=no_warm_positive_context. Mixed context ignores cold items in the CSR while retaining unique warm/cold counts. Duplicate context IDs are deduplicated for binary feedback. No fit/partial_fit or targets are used for fold-in.

Reproducibility is checked with fixed seed and one CPU/BLAS thread on the current environment; cross-platform bit-for-bit equality is not promised. There was no parameter selection or tuning based on these metrics.

## Serialization

Local bundle: artifacts/models/are006/ with native numeric ALS NPZ, item/user mapping Parquet, popularity Parquet, training_config.json and metadata.json. No pickle; installed native load uses allow_pickle=False. SHA-256 covers all five payload files; metadata records canonical mapping fingerprint, train-file hashes, dataset configuration, versions, factor sizes and fit times. Expected mapping and train fingerprint are checked before reusing a model. Artifacts are integrity-checked local files, not authenticated downloads from untrusted sources.

Reloaded recommendations were compared with the original models for all 100 validation users (same IDs and scores within rtol=1e-5, atol=1e-6). Running load-only does not fit again. Validation leaves both trained user factors and item factors exactly unchanged.

## Validation definitions

All 100 validation users are retained. Every user has warm positive context and positive targets in this snapshot. Synthetic tests additionally cover no warm context, cold-only context and no positive targets.

Both models rank the same train vocabulary minus ALL observed context items, including low/zero ratings. Only positive context is passed to ALS. Generation has no target argument. Context-target overlap is rejected. Target is used exclusively after recommendations are generated.

Recall@10 aliases Full Recall@10, with all positive targets including cold items in the denominator. NDCG@10 also uses full targets in its ideal denominator. Warm Recall@10 is conditional on nonempty warm context and warm target; it is not a whole-service metric. Users without positive targets remain in the cohort and have undefined Recall/NDCG, excluded only from those macro means with counts reported. ALS without warm context gives an empty list and zero Recall when positive targets exist. ALS diagnostics are ALS-only without fallback.

Catalog coverage is distinct recommended items / train vocabulary. Cold target share is cold positive target rows / all positive target rows. Test files are never read or checksummed in this task; a synthetic integration test corrupts test artifacts and verifies the training pipeline still runs.

## Actual local results

Fit wall time (fit only, not loading/serialization/diagnostics): {"als": 0.24102939999829687, "popularity": 0.008540400000129011} seconds.

Factor shapes: {"items": [2091, 64], "users": [772, 64]}.

| model_name | users | full_recall_users | warm_recall_users | full_recall_at_k | ndcg_at_k | warm_recall_at_k | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 100 | 100 | 100 | 0.212262 | 0.219222 | 0.215081 | 0.018651 |
| als | 100 | 100 | 100 | 0.28371 | 0.388166 | 0.291948 | 0.11143 |

Both models: 1,538 positive targets; 223 cold targets (14.4993%). No users without warm context or positive targets in this validation snapshot. Popularity recommends 39 distinct items, ALS 233.

### Top-10 popularity

| rank | title | dataset_anime_id | myanimelist_id | score |
| --- | --- | --- | --- | --- |
| 1 | My Hero Academia | 100 | 31964 | 338.0 |
| 2 | One Punch Man | 92 | 30276 | 315.0 |
| 3 | My Hero Academia Season 2 | 1167 | 33486 | 311.0 |
| 4 | Demon Slayer: Kimetsu no Yaiba | 2376 | 38000 | 293.0 |
| 5 | Your Name. | 1147 | 32281 | 276.0 |
| 6 | Hunter x Hunter | 726 | 11061 | 270.0 |
| 7 | My Hero Academia Season 3 | 1186 | 36456 | 264.0 |
| 8 | Attack on Titan Season 2 | 1160 | 25777 | 261.0 |
| 9 | Fullmetal Alchemist: Brotherhood | 99 | 5114 | 260.0 |
| 10 | Jujutsu Kaisen | 2384 | 40748 | 240.0 |

### Top-10 als

| rank | title | dataset_anime_id | myanimelist_id | score |
| --- | --- | --- | --- | --- |
| 1 | Clannad | 409 | 2167 | 0.747589 |
| 2 | OreImo | 583 | 8769 | 0.684091 |
| 3 | Elfen Lied | 275 | 226 | 0.609099 |
| 4 | Erased | 93 | 31043 | 0.581201 |
| 5 | Bakemonogatari | 115 | 5081 | 0.553329 |
| 6 | Fullmetal Alchemist: Brotherhood | 99 | 5114 | 0.531247 |
| 7 | Code Geass: Lelouch of the Rebellion | 144 | 1575 | 0.48566 |
| 8 | High School of the Dead | 54 | 8074 | 0.485646 |
| 9 | Hyouka | 160 | 12189 | 0.473591 |
| 10 | Charlotte | 1126 | 28999 | 0.468032 |

The example is the same validation user absent from train for both models. ALS scores are ranking dot products, not probabilities and not comparable to popularity counts.

## Checks and limitations

The full suite includes actual tiny ALS fits, native mathematical fold-in, cold/empty contexts, target independence, train-only popularity, deterministic ties, reload parity, mapping mismatch/corruption rejection, untouched test files and pipeline load-only without refitting. Existing tests cover three-way user disjointness from ARE-005.1.

99 tests passed in 3.00 seconds without pytest warnings. Notebook 04 executed all cells successfully in memory and is kept with no persisted outputs. The Windows Jupyter run emitted environment warnings about its event-loop selector helper and local TCP kernel transport; execution succeeded. Data, models and diagnostics remain under ignored directories. A scan of Git candidate files found no local token values, model/data artifacts or files larger than 1 MiB. All notebooks have empty outputs and execution counts. Production modules do not import Jupyter. No Git commit or push was performed.

Dependency issue: implicit 0.7.2 had no compatible Python 3.12 Windows wheel, and source installation required unavailable MSVC/nmake. implicit is pinned to 0.7.3, which provides the compatible official wheel; no compiler was installed. Relevant installed versions are recorded in model metadata:

{
  "implicit": "0.7.3",
  "numpy": "2.5.4",
  "pandas": "2.3.3",
  "pyarrow": "23.0.1",
  "scipy": "1.18.1",
  "threadpoolctl": "3.8.0"
}

The current sample is small; a random context/target split without timestamps measures held-out preference reconstruction, not future chronological performance. Popularity/ALS differences here do not establish significance or production effectiveness. Binary confidence=1 and fixed initial hyperparameters are intentionally untuned. Training and YummyAnime signals differ; mapping and the documented future status/favourite policy must be applied in a separately authorized online integration. No automatic fallback is included in these ALS numbers. Full experiment evaluation belongs to ARE-007.


## Interactive Notebook 04 follow-up

New reusable modules:

- preprocessing/profile_signals.py: normalize list/favorite union, preserve original fields, binary positive policy, MAL mapping, warm positives, full seen set and coverage diagnostics.
- recommendation_exploration.py: explicit model loading/training, validation selection, title/popularity enrichment, post-generation hit annotations, existing metric calls, score vectors and optional profile inference. No Jupyter imports or profile persistence.
- tests/test_recommendation_exploration.py: synthetic histories/models and mocked API; no network calls.
- matplotlib is added to the research extra for notebook plots.

Parameters are in one initial cell: factors, regularization, iterations, seed, retrain, top_k, validation_user_id and optional yummyanime_nickname. The saved ALS configuration is checked before reuse. Missing/incompatible models raise an explicit retraining message rather than silently fitting. Changed model configurations use separate ignored *-notebook artifact directories. No automatic tuning. top_k/user_id do not enter model fitting/loading parameters. Rerun only the selected-user section for a different user, and the cohort diagnostic section when K changes.

The selected user's observed context includes titles, unchanged rating/rating_raw, positive and warm flags. Targets are shown separately. Both models generate recommendations without any target argument; only afterwards are is_hit and metrics computed. Existing diagnostic definitions and all-context seen filtering are reused. Plots show cohort Full Recall/NDCG, the selected user's full-vocabulary ALS scores, train popularity per recommendation rank, and per-model target hit counts.

The optional YummyAnime section defaults to disabled and makes zero requests when the nickname is blank. It uses the existing public-token client, stopping on API errors. A successful profile uses three GETs (profile, lists, favorites) without retries or Bearer authorization. Lists retain their statuses; favorites mark membership and add favorite-only rows. Invalid raw rating values are distinguished from genuinely missing values before completed/unrated labeling. Binary positives are valid rating>=5, completed with zero/missing rating, or favorite. Planned/dropped are a veto even if rated/favorite. Cold/unmapped positives are counted, but only distinct warm positives enter a unit-confidence fold-in. All mapped history is excluded from both ALS and the explicit Popularity fallback. An API failure does not cause fallback on a partial history.

Mapping coverage is reported separately against the full local catalog and the actual train vocabulary, over valid unique MAL IDs. Missing/unmapped IDs cannot be used to identify and exclude items; counts make this limitation visible. The raw profile and lists are never saved. No profile nickname or personal list entries are included in this report.

Live smoke check on the authorized own-profile snapshot: the sandbox attempt failed at the first network request; a permitted outside-sandbox check then returned HTTP 200 for all three endpoints. 72 unique MAL-linked history records, 66 catalog matches (91.67%), 50 train-vocabulary matches (69.44%); 50 positive rows, 35 distinct warm positives and 15 cold/unmapped positive rows. ALS returned 10 titled recommendations, none from mapped known history. No fallback was needed; item factors remained exactly unchanged. These counts can change as the profile/catalog changes and are not a historical completeness guarantee.

Notebook 04 executed all cells including plots in memory with no persisted outputs. Tests cover a second user, changed K, target independence, preserved ratings/titles, explicit retraining, favorites-only entries, duplicate and ambiguous mappings, planned/dropped veto, empty/cold fallback, no fitting during profile inference and stop-on-auth-error. The training dataset is unchanged. The adapter is experimental: status/favorite inference signals differ from rating-only training data and are not evidence of recommendation quality.

Final follow-up checks: 107 tests passed in 4.16 seconds. Two actual validation users were checked with K=5 and K=7: both models returned the requested counts with valid titles and no known items. Changing factors to an untrained value was rejected without fitting. All notebook outputs/execution counts remain empty, and the Git-candidate scan found no local token values, data/model artifacts or large files. No commit or push.

## Publication audit

The user authorized publication of ARE-006 to First-light-4/anime-recommendation-engine,
branch main. Final tests: 107 passed in 3.40 seconds. The review confirmed separate
training, inference, signal preparation and diagnostic responsibilities, shared ranking
and mapping logic, and no business-logic implementations in Notebook 04. Targets are
not passed to recommendation generation or fold-in. Public profile responses remain
in memory; their histories and tokens are not publication artifacts. No ARE-007 work
was started. Git remote and authenticated account were verified before publication.
