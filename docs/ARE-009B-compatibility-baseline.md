# ARE-009B — Architecture compatibility baseline

Migration note ARE-009C: [contracts and registry](ARE-009C-contracts-and-registry.md)
documents the new neutral implementations and compatibility adapters. This file
remains the pre-migration baseline. The intentional writer policy change is
immutable publication: `save_models` now rejects every existing destination,
including an empty directory; readers, serialized v1 bundles and predictions
retain compatibility.

Статус: compatibility baseline для review перед ARE-009C/009D. Дата: 24 ноября
2026. Production imports, алгоритмы, configs, models, datasets и Notebook не
изменялись. Исправлены только подтверждённые ошибки README/ARE-008 и добавлены
regression tests. ARE-009A остаётся историческим аудитом, его findings не переписаны.

## 1. Scope и использование при миграции

Baseline описывает **существующий** Python API и artifacts, а не предлагаемую
архитектуру ARE-009A. Он не обещает, что нынешние CLI независимы от соседних stages.
Миграция должна сохранить записи старых formats, ID semantics, predictions/metrics
и side effects каждого публичного сценария. Новый нейтральный API может появиться
параллельно; старые imports должны иметь adapters до отдельного удаления.

Serving default — **10k**, top_k=15. 1k/5k/10k aliases и их текущие пути сохранены.
Frozen training/evaluation configs ARE-006/007A по-прежнему относятся к 1k и не
переименовываются в serving defaults. Notebook 01 защищён независимо от остальных.

## 2. Публичные интерфейсы и эффекты

Пути в таблице относительны `src/anime_recommender/`. Methods Python classes
считаются частью контракта; изменение организации файлов не разрешает молча
сменить их семантику. DataFrame source inputs не должны мутировать.

| Компонент / public API | Обязательные inputs | Outputs / format | Эффекты / обязательные guards |
| --- | --- | --- | --- |
| `ingestion.yummyanime.YummyAnimeClient`; `get_user/get_lists/get_favorites/get_rating_statistics` | public application token; nickname/positive user ID | validated public records, sanitized request_log; `YummyAnimeError(status,retry_after)` / authorization/network subclasses | HTTP только при явных методах; X-Application, без private/Bearer, redirects/retries/response-body logging; no persistence |
| `ingestion.processed_data.load_processed_data(directory)` | ARE-004 directory, successful quality report, clean/catalog/mapping files | `ProcessedData(interactions,catalog,mapping,quality_report)`; SHA verified | local read only; corrupt/failed snapshot raises, не download/rebuild |
| `preprocessing.training_data.prepare_catalog(raw)` | animeID/title/mal_url source columns | clean catalog, one-to-one dataset↔MAL mapping, quarantine, report | no IO; MAL из URL, не внутренний animeID; ambiguous MAL исключается |
| `clean_interactions(raw,mapping,rating_min=0,rating_max=10)` | userID/animeID/rating + certified map | sorted unique user-item clean frame с rating/rating_raw/MAL, accounting report | no IO; exact duplicates убрать; conflicting pairs drop; zero сохранить; raw=clean+sum(removals) |
| `positive_feedback(interactions,threshold=5,confidence=1)` | clean unique pairs; valid ratings | positives с неизменными rating/rating_raw, `positive=1`, float32 `confidence=1` | no IO; zero/missing не positives; configurable 5/6/7/8, основной5 |
| `data_split.assign_users`, `split_observed_interactions`, `filter_train` | raw IDs/history, seed42, configured fractions/minima | disjoint assignments; raw context/target; train-only k-core + rounds | no IO; holdout до positive filtering; eval activity selection отсутствует |
| `sparse_matrix.build_train_matrix`, `build_context_matrix` | train positives + MAL mapping / context+cohort+train items | CSR, ordered row/column mappings | no IO; orientation users×items; no duplicate confidence summation; only warm context |
| `training.dataset.prepare_dataset` / `load_dataset` | clean source/map/config; optional fixed assignments AND holdouts / v2 directory | `PreparedDataset(tables,matrix,report,context_matrices)` | prepare pure; full loader читает и validation, и test по dataset contract; не использовать как selective evaluation reader |
| `training.inputs.load_training_inputs(directory)` | certified v2 dataset, optional verified shared/baseline refs | `TrainingInputs` с train+validation и fingerprints | read-only; только 8 train/validation files, никогда test; нынешнее объединение views сохраняется до отдельной миграции |
| `training.artifacts.load_models(path,expected_mapping=None,expected_train_fingerprint=None)` | self-contained six-file v1 bundle | `ModelBundle(als,popularity,items,train_users,metadata)` | SHA/config/mapping/shape/fingerprint checks; native load без pickle, no fit/download; legacy1k без model_version допустим |
| `ModelBundle.recommenders()` | готовый bundle | dict `popularity`, `als` | создаёт inference wrappers, не обучает модели |
| `inference.model_versions.load_version_registry(root)`, `load_version_workspace(root,model_version=None)` | current registry, saved bundle, certified catalog, local metadata | registry dict / `ModelWorkspace(bundle,inputs=None,catalog,franchise_config,franchise_mapping)` | omitted version → registry10k; exact aliases1k/5k/10k; missing/corrupt selected version => error, без silent version fallback |
| `inference.fold_in.fold_in(model,user_context,item_mapping)` | native saved ALS; positive **dataset IDs**; ordered matching items | `FoldInResult(available,factor,context_vector,diagnostics)` | fixed-item solve `recalculate_user`; no fit/partial_fit, no factor mutation; cold-only/empty => unavailable, factor=None |
| `ALSRecommender.recommend` / `PopularityRecommender.recommend` | context, top_k, optional `seen_items`, `exclude_seen=True` | `RecommendationResult(recommendations,available,diagnostics,reason)` | candidates train vocabulary; positives/observed-known исключены; no target input; no network/IO |
| `inference.recommendations.rank_scores` | finite aligned score vector/map/model_name/seen/K | core five-column DataFrame | score desc, dataset ID asc tie-break; contiguous ranks; K positive int; no duplicate/unknown recommendations |
| `preprocessing.profile_signals.normalize_profile_history`, `prepare_profile_signals` | API lists/favorites / normalized history+catalog+items | normalized source / `ProfileSignals(frame,positive_ids,seen_ids,coverage)` | source не изменяется; MAL join1:1; completed-unrated OR rating>=5 OR favorite; planned/dropped veto; бинарно, без новых весов |
| `inference.profile_session.ProfileSession.workspace`, `.recommend`, `.initial_titles` | root, optional client factory; nickname/version/K; memory cache | model workspace; profile dict; optional enriched titles | once-per-nickname fetch(profile/lists/favorites), per-model mapping; model/K switch no fetch/fit/rebuild; initial bounded title lookup1 раз, не franchise discovery |
| `profile_display(profile,nickname,show_history=False)` | session result | summary Series, recommendation table, optional history | presentation only; default скрывает private history, не сохраняет её |
| `inference.franchise_filter.filter_candidates`, `recommend_with_franchise` | candidates / recommender; local structured graph; completed/seen IDs; model vocabulary | `FranchiseResult(recommendations,diagnostics,audit)` | pure filtering/local pool expansion; no HTTP/fit; no invented title links; factors/scores не изменяются |
| `preprocessing.franchise_mapping.load_franchise_resources` | root/config, optional prepared/static metadata | config + validated graph + explicit warning | local-only; invalid/absent metadata → safe partial/empty mapping, не online discovery |
| `ranking_metrics` and scalar wrappers | ordered unique recommendation IDs, targets, K | precision/recall/ndcg/map/hit_rate scalars | empty relevant set => None, empty list with targets =>0; no IO |
| `ranking_evaluator.evaluate_models(bundle,inputs,ks,include_als_fallback)` | saved bundle + selective validation; compatible train/map IDs | `EvaluationReport(metadata,tables)` | validation only; no fit, target only after recommendation generation; factor immutability |
| `recommendation_checks.evaluate_validation` / `diagnostic_metrics` | same core frames/statuses; legacy callers | legacy summary/per_user/recs/status | Full metrics те же; Warm legacy eligibility отличается, см. раздел4 |
| `evaluation.reports.save_report`, `load_report` | EvaluationReport / directory | JSON manifest + named Parquet tables / verified report | save пишет только выбранный report directory; load проверяет SHA и JSON-vs-Parquet, no fit/network |
| Notebook05 caller | model/certified catalog; synthetic/real fetched profile; local franchise graph | Top15, diagnostics; source cached only in kernel | default10k; 1k→10k→5k→1k и K change no refetch/fit; unknown metadata warning; no outputs в Git |

Некоторые старые runner APIs **явно небезопасны для load-only вызова**:
`training.pipeline.run_experiment(train=True)` и `training.dataset.run_pipeline`
могут писать файлы, а fit runner — перезаписывать модель. Их нельзя подставить
вместо load_models/ProfileSession. ARE-009B не меняет эти effects и не скрывает
architectural coupling, обнаруженный в ARE-009A.

## 3. Output schemas и identities

### Таблицы и numerical contracts

- Clean interactions: `dataset_user_id`, `dataset_anime_id`, `rating` float64,
  `rating_raw` string, `myanimelist_id`; одна строка на пару, stable sorted IDs.
- Model item map: `dataset_anime_id`, `myanimelist_id`, `matrix_col`; unique
  positive int64 IDs, matrix_col=0…N−1, row order соответствует item factors.
- User map: `dataset_user_id`, `matrix_row`=0…U−1; aliases не меняют dataset IDs.
- Train CSR float32, sorted indices, shape(U,I), values1 и nnz=len(train positives).
  Fold-in context CSR shape(1,I); cold IDs в ней отсутствуют.
- Core recommendations строго: `dataset_anime_id`, `myanimelist_id`, `score`,
  `rank`, `model_name`. Scores конечные dot products/counts, не probabilities.
  Empty/cold-only ALS context возвращает available=False и
  reason="no_warm_positive_context"; application service может выбрать Popularity
  только внутри той же версии после успешного получения профиля. Model missing
  или API error не превращаются в другой model_version/fallback по ошибке.
- Profile display: `Rank`, `English Title`, `Russian Title`, `ALS Score`, `MAL ID`.
  При отсутствии подтверждённого русского названия — `—`, не перевод. English
  fallback может быть romanized catalog title. При Popularity ALS Score — `—`.
- `top_k` не является параметром обучения. Franchise restrictions/candidate
  exhaustion могут дать меньше K; forbidden items не возвращаются для заполнения.
- Only completed/list_id=2 доказывает просмотр для последовательности. Planned/
  watching исключаются из повтора, но не подтверждают прохождение сезона.

### Форматы artifacts

| Artifact | Текущий поддерживаемый контракт | Что не менять при извлечении кода |
| --- | --- | --- |
| ARE-004 processed | quality report status=success, pipeline_version=1, output_sha256, Parquet tables | conservative cleanup/accounting, MAL identity, raw ratings |
| ARE-005.1 dataset | dataset_version=2, raw_holdout_then_positive_v2, artifact_sha256, user assignments, train/eval CSR/tables | raw context/target ID/rating content, empty eval users, train-only vocab, metadata/source hashes |
| Scaling | fixed-evaluation-bottom-k-v1, dataset v2; sampling manifest; optional baseline_reference/shared_artifacts | baseline inclusion, nested train, absolute legacy refs читаются; changing cold flags допустимо только от train vocabulary |
| Model bundle | format_version=1, native ALS NPZ + item/user/popularity Parquet + training_config.json + metadata.json | шесть файлов, contiguous mappings, native factor shapes, file/config/map/train checks; не перепаковывать реальные bundles |
| Evaluation | ARE-007A-v1 metadata; 11 named Parquet tables + report.json с SHA/JSON aggregates | Full/Warm policy, eligible counts, protocol/dataset/model identities; no target-derived recommendations |
| Legacy diagnostics | ARE-006 JSON/Parquet; ARE-007C separate filtered-vs-original reports | legacy Warm labels/results не переписывать в canonical задним числом |
| Metadata | schema_version=1 latest pointer/checkpoint/source snapshot; immutable version manifest and four payload files | graph provenance/roles/barriers, 502 retryability, no title inference |

Текущий model loader не делает отдельный strict dispatch по format_version;
v1 — подтверждённый совместимый формат, не обещание поддержки неизвестного v2.
Это известная граница ARE-009A, а не повод поменять loader в этой задаче.

Fingerprint families — разные контракты, не взаимозаменяемые SHA:

1. `file_sha256`: digest **байтов файла**; сериализатор/перенос может менять его.
2. `mapping_fingerprint`: ordered int64 rows validated map → compact JSON → SHA.
3. Train fingerprint: SHA JSON `{train_hashes,dataset_config}` со старым sort_keys/
   separators behavior. В hash входят четыре train файла, не targets.
4. `evaluation_fingerprint`: train identity + stable sorted validation user/context/
   observed/target frames, JSON precision15. Между vocabularies identity различается.
5. `model_fingerprint`: map/train/config/popularity + factor dtype/shape/bytes;
   отражает загруженное содержимое модели.
6. Fixed-validation identity: raw context/positive context/targets без cold flags,
   плюс fixed user IDs. Для 1k/5k/10k:
   `d01f4dfe46d178ab9a6c721a24e262de76c96e2939a31178b9ea65dda0c4b63e`.

При переносе helpers сохранять существующую canonicalization, exception identity
и file bytes. Пересчёт новым удобным JSON helper — потенциальное breaking change.
Cross-platform/новые library versions не обязаны давать bit-identical Parquet/ALS.

Model locations: 1k=`artifacts/models/are006`,
5k=`artifacts/models/are008b/5k`, 10k=`artifacts/models/are008b/10k`.
Ни models, ни datasets не в Git. Требуются свой trusted local bundle+catalog
или явное [воспроизведение](ARE-008-artifacts-and-reproduction.md).
Notebook не делает это автоматически. Source full scan в ARE-009B не выполнялся.

## 4. Canonical Warm eligibility

Пусть `R_u` — все positive targets пользователя, `V` — train vocabulary,
`Rwarm_u=R_u∩V`, `Cwarm_u` — warm positive context.

| Условие | Legacy ARE-006/diagnostic Warm Recall | Canonical ARE-007A-v1 Warm |
| --- | --- | --- |
| Eligible | `len(Rwarm_u)>0 AND len(Cwarm_u)>0` | **только `len(Rwarm_u)>0`** |
| Warm target есть, warm context нет | excluded/undefined | included; unavailable ALS получает empty list и metric=0 |
| Warm targets нет | undefined | undefined |
| Нет positive targets | Full и Warm undefined | Full и Warm undefined |
| Что публикуется | warm_recall_at_k | пять metrics в `mode="warm"`, eligible counts |

**Canonical для дальнейших ARE-009C/D: ARE-007A-v1**, уже используемый main
evaluation и ARE-008B. На этом этапе код не меняется. Для Warm берётся тот же
замороженный recommendation list, relevant set — только warm targets; дополнительного
warm-context фильтра нет. ALS-only и ALS-with-Popularity-fallback различаются
отдельными labels, наличие fallback не меняет eligibility.

Full включает cold targets в Recall/NDCG denominators. Precision делится на K,
даже если список короче; MAP нормируется min(K, relevant_count); NDCG ideal также
min(K, relevant_count). Macro — среднее по defined users; undefined не превращать
в ноль. Cohorts задаются context, не target. Изменять эти правила в рефакторинге нельзя.

### Callers и ранее опубликованные результаты

| Отчёты / UI | Реализация / protocol |
| --- | --- |
| ARE-006 `artifacts/validation/are006`, Notebook04 `validation_diagnostics/view` | legacy diagnostic_metrics; Warm requires context+targets |
| ARE-007C `artifacts/evaluation/are007c/<stamp>` | отдельная franchise диагностика через diagnostic_metrics, legacy Warm |
| ARE-007A `artifacts/evaluation/are007a/<stamp>`, Notebook06 | canonical evaluate_models, ARE-007A-v1 |
| ARE-008B `artifacts/evaluation/are008b/{1k,5k,10k}`, Notebook07 quality plots | canonical evaluate_models; validation_view для примера legacy diagnostics, но не источник aggregate comparison metrics |
| ARE-008B.1 bootstrap | сохранённые canonical **Full** metrics; Warm eligibility на bootstrap не влияет |
| Dataset user_mapping.warm_recall_eligible и evaluation_summary.warm_recall_users | legacy diagnostic flags/counts, **не authoritative canonical eligibility** |

Проверены existing local reports: ARE-006, все пять ARE-007A snapshots, три
ARE-007C snapshots; no-warm-context=0 и Warm eligible=100. New saved-validation
replay для каждой 1k/5k/10k дополнительно проверяет: все 100 users имеют warm context
и nonempty warm targets, legacy/canonical Warm совпадают (abs tolerance1e-12).
**На опубликованные значения этих сохранённых cohorts различие не повлияло.**
Full metrics семантически не зависят от Warm mask вообще. Для произвольного
будущего cold-context cohort совпадение не обещается.

Контрпример из synthetic regression: Warm target есть у3 users, warm context у1;
ALS hit уэтого user. Legacy Warm Recall=1 с eligible1; canonical=1/3 с eligible3.
Это зафиксированное различие поведения, не исправленный алгоритм. Второй тест
подтверждает, что canonical игнорирует legacy dataset eligibility flag.

## 5. Regression test matrix и numerical accuracy

| Инвариант | Тесты / подтверждение |
| --- | --- |
| Cleaning/accounting/MAL identity/raw ratings | test_training_data_pipeline, test_training_dataset_research |
| Raw user-disjoint holdout до positives; no leakage | test_ml_dataset_preparation, test_scaling |
| Nested samples / fixed context-target / warm-only CSR | test_scaling, test_ml_dataset_preparation |
| Default10k, aliases, mappings, missing/corrupt/config/hash errors | test_model_versions; новый saved-validation replay |
| Legacy1k bundle без model_version загружается | новый `test_compatibility_baseline.py`, expected format_version1 и expected fingerprint |
| No network/download/fit/ingestion во время replay | patches в test_compatibility_baseline и test_model_versions |
| Fold-in exact solve, orientation, no factor mutation | test_recommendation_models; новый replay проверяет loaded arrays before/after |
| Candidate set/seen exclusion/target independence | test_recommendation_models, test_ranking_evaluation, test_recommendation_exploration |
| Unknown franchise metadata не выдумывается; completed-only evidence/scores | test_franchise_filter, test_yummy_viewing_order, test_viewing_order_metadata_pipeline |
| Full/Warm/undefined/macro/MAP math | test_ranking_evaluation, test_model_selection |
| Legacy vs canonical Warm явно различаются | 2 новых tests в test_ranking_evaluation |
| Saved metrics/predictions 1k/5k/10k воспроизводятся | 3 parametrized cases test_compatibility_baseline |
| Notebook05 actual bundles + synthetic profile + all-version switching | новый test_notebook05_compatibility, no HTTP/fit/ingestion/source/test reads |
| README default/UTF-8 docs | точечная правка + static consistency/link checks; configs/Notebook неизменны |

Replay tolerance (это compatibility gate, не разрешение изменить ranking):

- Recommendation IDs, model labels, ranks и MAL IDs — **точное** совпадение.
- Сохранённые aggregate metrics: atol=rtol=1e-10.
- Scores: atol=1e-6, rtol=1e-5 (native numerical backend).
- Published rounded six-digit means: abs tolerance5e-7, rel=0.
- Legacy/canonical Warm при identical eligible cohort: abs tolerance1e-12.
- Mapping/model/config/train fingerprints и file hashes — точное совпадение.

| Version | Train U×I / nnz | Full Recall@10 | Full NDCG@10 | Warm Recall@10 |
| --- | --- | ---: | ---: | ---: |
| 1k | 772×2091 / 48080 | 0.283710 | 0.388166 | 0.291948 |
| 5k | 4621×5279 / 359967 | 0.301667 | 0.423164 | 0.302479 |
| 10k | 9455×6719 / 745354 | 0.305145 | 0.425284 | 0.305863 |

Integration fixtures локальные/ignored и опциональны на clean clone. Если bundles/
datasets/reports отсутствуют, эти cases **skip с причиной**, а не download/rebuild.
Synthetic tests остаются обязательными. Версии библиотек/платформы могут изменить
prediction ranking при новом обучении: такая смена не объявляется совместимой
просто из-за широкого numeric tolerance; нужен отдельный review нового baseline.

## 6. Готовность к дальнейшему рефакторингу и ограничения

Следующие компоненты имеют достаточные behavioral guards для постепенного extraction:
hash/atomic helpers, existing shared records/errors, model read/registry facade,
MAL matching, pure fold-in/rank/seen filter, franchise rules, selective dataset
readers и evaluation wrappers. Перенос разрешается только отдельными ARE-009C/D.
Нельзя одновременно с relocation менять serialized bytes, scores, eligibility,
source sampling, thresholds, registry default или незаметно расширять API calls.

Сохраняются известные проблемы ARE-009A: transitive inference→research→training
imports, default-fit runner/overwrite behavior, absolute shared dataset references,
hardcoded three aliases, incomplete artifact format dispatch. Tests фиксируют
совместимость, но не объявляют эти границы уже исправленными.

Выбор10k — принятое product decision, не доказанное превосходство над5k. Validation
только100users, bootstrap pointwise/conditional, 99% exposure в top10% vocabulary,
franchise coverage10k около0.40%; неизвестные relationships остаются неизвестными.
Не использовать test для повторного model selection. Main reported ALS metrics
без franchise filter не обещают такие же значения для filtered Top15.

В ARE-009B модели/данные не восстанавливались заново, реальные API не вызывались.
Полный pytest и notebook/replay checks используют existing local validation или
малые synthetic fixtures. Existing synthetic training tests могут создавать tiny
models в temporary directories, но не переобучают сохранённые real1k/5k/10k.
Notebook01 и его user edit сохраняются побайтно. Результаты финальной SHA-сверки,
pytest и smoke приводятся ниже после выполнения.

## 7. Результаты приёмочных проверок ARE-009B

- Полный запуск `python -m pytest -q -p no:cacheprovider`: **210 passed**,
  1 warning, **34.67 с**; skipped=0 в текущем local environment.
- Добавлено 8 regression cases: 3 saved-validation replays, 2 Warm differences/
  legacy-flag invariants, 2 docs consistency/encoding/link checks, 1 Notebook smoke.
- Отдельный запуск Notebook05 smoke: **1 passed**, 4.98 с. Notebook исполнялся
  только в памяти, с реальными 1k/5k/10k и synthetic profile, затем переключением
  1k→10k→5k→1k. Проверены default10k, Top15/Top7, mapping, seen exclusion, source
  cache, franchise enabled и partial warning. Каждый mock profile endpoint вызван
  один раз; initial title helper — один раз. Реальная сеть/fit/ingestion/source
  interaction/test reads в этом сценарии запрещены; outputs не сохранены.
- Saved metrics, recommendation IDs/ranks и factors прошли tolerances раздела5.
  До/после совпали SHA model/evaluation files, replay не пишет reports.
- Warning — штатный Windows Proactor/zmq fallback selector thread; ошибок ячеек нет.
- Контрольный snapshot содержал359файлов. Изменены только четыре разрешённых:
  README, ARE-008B report, ARE-008B.1 report, test_ranking_evaluation.py;
  **355 остальных SHA совпали**, включая source/configs/datasets/models/metadata,
  прежний ARE-009A audit, diagnostic script и все Notebook. Добавлены только
  новый compatibility manifest и три test modules.
- Исправление docs не меняет ни одного ранее опубликованного численного результата.
  Выявленная несовместимость — eligibility для cold-context users; она сохранена
  как два явно описанных historical/canonical контракта, не замаскирована правкой.
- Рабочая основа для отдельной задачи ARE-009C готова. Архитектурная миграция не
  начата; staging пуст, commit/push не выполнялись.
