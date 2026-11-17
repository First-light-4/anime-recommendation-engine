# ARE-007C: franchise-aware recommendation post-processing

**Update, 17 November 2026:** this initial AniList-based report is retained as the
history of the first configuration. The earlier research missed YummyAnime's
`viewing_order` field. The active source is now the structured YummyAnime order,
with generated mainline progression and 22 covered train items across three
families. See [the API investigation and updated policy](ARE-007C-yummy-viewing-order.md).
The initial metrics/configuration below must not be interpreted as the current
source coverage.

Implemented locally; awaiting code review. No commit/push, retraining, threshold/split changes, dataset enlargement or ARE-008 work.

## Diagnosis before implementation

Existing inference excludes observed dataset IDs obtained by exact MAL mapping, irrespective of rating/status. It correctly excludes an exact known MAL item when the mapping exists. It does not infer that earlier seasons were consumed merely because a later season is present. Thus a missing earlier season in a user's list remains a valid independent item for ALS. ALS ranks items without franchise structure, so multiple seasons can appear together.

Missing/nonpositive MAL IDs cannot be mapped to train items; the existing adapter reports that coverage gap and does not guess by title. The new post-filter additionally checks raw observed MAL IDs directly against candidate MAL IDs, without relying solely on mapped dataset IDs. Planned/watching/dropped/favorite-only parts are all excluded when explicitly observed, but only list_id=2 is completion evidence. A positive rating is not converted into a completed status.

The local catalog has only a boolean sequel flag, titles and metadata; it has no trustworthy prequel/sequel ID edges. A boolean is insufficient to derive a franchise or watch order. The official YummyAnime Swagger describes /anime/{id}/recommendations as similar anime, not a typed prequel graph; its season field is calendar season. No undocumented relationship semantics or title similarity are used.

## Sources and bounded scope

- [YummyAnime Swagger](https://ru.yummyani.me/api/swagger): existing profile and title metadata integration; no reliable typed franchise graph established there.
- [AniList relations query](https://docs.anilist.co/guide/graphql/queries/media) and [relation types](https://docs.anilist.co/reference/enum/mediarelation): one bounded query for 15 explicitly selected Attack on Titan / Demon Slayer entries, HTTP 200. It returned 31 identified anime nodes and 50 directed typed records (including reciprocal relations), not a full catalog. No recursion, pagination or further relation queries from inference.
- [AniList terms](https://docs.anilist.co/guide/terms-of-use) and [rate limits](https://docs.anilist.co/guide/rate-limiting) were reviewed. This is a small noncommercial complementary research use, not a competing tracker, bulk collection or an assertion of an unrestricted license to all provider data. A broader collection is outside authorization.
- A single initial Jikan relation probe timed out; no Jikan data was used. No parser or authorization/rate-limit bypass was implemented.

configs/franchise_relations.json retains only public identifiers, formats, typed edges, queried/frontier flags, source URL, retrieval time and an explicit reviewed policy. The raw bounded metadata response is local under ignored data/external/. No user profiles are included in that snapshot.

There are two connected known families; CHARACTER/OTHER/ADAPTATION do not automatically establish a franchise. Franchise IDs are generated as mal-family-<component-minimum-MAL-ID>, versioned by the snapshot fingerprint, not official external franchise IDs. Unknown items remain independent rather than being grouped by name.

## Mainline policy and nonlinearity

Structured source relation labels are not a universal mandatory watch-order specification. The filtering policy therefore separately marks only the reviewed 14 main-story entries for the two case studies, with entry MAL IDs 16498 and 38000. It does not infer mainline membership from titles. All source edges remain available for inspection.

For example, No Regrets (25781) has a PREQUEL relation to Attack on Titan in the source, but this optional OVA is deliberately outside the required TV mainline. Completing a late TV season does not imply the OVA, films or spin-offs were watched. SUMMARY, SIDE_STORY, PARENT, SPIN_OFF and ALTERNATIVE establish family membership where applicable, not automatic viewing or mandatory predecessors.

Within the reviewed mainline, the Demon Slayer Mugen Train movie (40456) and TV version (49926) are explicit alternative prerequisite branches. Either completed branch supports entry into 47778. This does not mark the other version itself as watched. An ordered path is returned as groups, for example ((38000,), (40456,49926), (47778,)), not as a single forced linear franchise list. Ambiguous multiple predecessor groups or incomplete paths return unknown order; no hard chronology exclusions are invented. Cyclic/contradictory snapshots are rejected.

Completing a confirmed later mainline part supports withholding its earlier mainline ancestor groups. It does not mark the full franchise consumed. All prerequisite groups on a trustworthy path must be covered by actual completed entries or their confirmed mainline ancestry before a late candidate is eligible. Unknown/nonlinear parts do not get arbitrary prerequisites.

## Files and responsibilities

```text
configs/franchise_filter.json              enabled, max-per-family=1, initial pool multiplier=3
configs/franchise_relations.json           bounded factual snapshot and reviewed policy
configs/franchise_evaluation.json          separate validation diagnostic configuration
src/anime_recommender/
  ingestion/anime_relations.py             opt-in bounded fetch and normalization; never inference
  preprocessing/franchise_mapping.py      components, partial order, alternatives, viewing_path
  inference/franchise_filter.py           pure filtering and local candidate expansion
  evaluation/franchise_evaluation.py      before/after validation diagnostics only
  recommendation_exploration.py          existing profile pipeline integrates the filter
notebooks/05_yummyanime_recommendations.ipynb
 tests/test_franchise_filter.py
 tests/test_recommendation_exploration.py
 tests/test_ranking_evaluation.py
README.md
 docs/ARE-007C-franchise-filtering.md
```

No business logic is in the notebook. The old ARE-007A evaluator/config/reports are not changed; the shared tests only add regressions. Notebook 01's existing user edit is untouched.

## Inference and diagnostics

Load a static mapping once with the saved profile workspace. The existing ALS or Popularity recommender generates a score-sorted pool of 3*N candidates with exclude_seen=False, so the post-filter can explicitly count observed exclusions. It then removes observed MAL/dataset IDs, confirmed earlier mainline groups, unavailable prerequisites and over-quota known families. Remaining scores and relative ordering are unchanged; final ranks are renumbered. If short, double the pool up to the full train vocabulary, never reinsert forbidden candidates. No API calls or model fitting occur in filtering/pool expansion. The existing no-warm-positive Popularity fallback remains explicit and receives identical franchise restrictions.

Unknown relationships are counted among the considered pool; unknown_final_recommendations counts retained unsupported items. Exclusion reasons are mutually exclusive; unknown status can overlap them. Other diagnostics include candidate count, observed/earlier/prerequisite/family exclusions, attempts, pool exhaustion, final count and mapping coverage. An exhausted pool may yield fewer than N, which is preferable to reintroducing a forbidden item. Disabling enabled in configs/franchise_filter.json restores original profile ranking; the separate filtering diagnostic requires it explicitly enabled.

Key APIs: build_franchise_mapping(snapshot), mapping.ancestors(mal_id), mapping.viewing_path(mal_id), filter_candidates(...), recommend_with_franchise(...), and run_franchise_evaluation(root). Existing profile refresh and title enrichment are reused. Changing K does not fetch new profile/relationship data; the existing optional title lookup may request newly selected MAL IDs.

## Validation diagnostic and limitations

```powershell
.venv\Scripts\python.exe -m anime_recommender.evaluation.franchise_evaluation
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Only validation is used. Output is saved to unique timestamp directories in ignored artifacts/evaluation/are007c/ as summary, per-user diagnostics, original/filtered recommendations, mapping coverage Parquet and a JSON manifest with model/dataset/relation fingerprints and file checksums. No old ARE-007A report is overwritten.

The rating-only validation data has no reliable completed status. Therefore completed_mal_ids=[] is used explicitly: ratings are NOT converted to watched evidence and targets never determine readiness. This conservative no-completion-proof scenario is not a full online YummyAnime effectiveness estimate; removing subsequent-season targets can mechanically lower offline Recall/NDCG. It is a separate post-processing configuration, not a replacement for the previous metrics.

| model_name | users | full_recall_at_k | ndcg_at_k | recommended_items | catalog_coverage |
| --- | --- | --- | --- | --- | --- |
| als_original | 100 | 0.283710 | 0.388166 | 233 | 0.111430 |
| als_franchise | 100 | 0.252278 | 0.353445 | 236 | 0.112865 |

Mapping coverage: 20/2091 (0.9565%) train items have known family membership; 13 have a reviewed trustworthy mainline order. Removed 103/1000 original Top-10 entries (10.3%); replacements are lower-ranked allowed candidates.

The loss of Recall/NDCG is reported without tuning or hiding it. Coverage is extremely partial (two selected franchises); unknown franchises can still occupy multiple Top-N slots or contain unrecognized late seasons. No complete-system franchise/diversity guarantee is claimed. Mainline review is filtering policy supported by explicit structured evidence, not an official universal watch order.

## Verification

152 tests passed in 6.07 seconds. Synthetic tests cover exact MAL/dataset exclusion, completed first/late seasons, planned-only history, empty context and fallback, optional movie/spin-off survival, missing/unknown IDs with similar names, alternative paths, cyclic/ambiguous graphs, family caps, pool expansion/exhaustion, unchanged scores/order and no network inside post-processing. Real static snapshot tests verify Attack on Titan and Demon Slayer branches. Integration tests corrupt test artifacts, forbid network/refitting, verify target mutation leaves predictions unchanged, and confirm an existing ARE-007A report checksum remains unchanged. Disabling the filter restores the original inference route and clears stale franchise diagnostics.

Notebook 05 completed all cells against the authorized own profile with fit and network relation-fetch requests forbidden. Three profile/list/favorite calls returned HTTP 200; existing bounded title enrichment was reused. The current snapshot produced Top-15 with no exact observed MAL IDs, at most one retained entry per known family and unchanged model factors/files. In its 45-candidate pool, 28 observed entries and one earlier mainline entry were excluded; no family-duplicate or prerequisite exclusion was needed in that particular pool. All 15 retained recommendations had unknown relationships in the small snapshot, so this live check does not prove diversity for other franchises. Train MAL mapping coverage was 71.05%, with 38 warm positive fold-in interactions. These are transient aggregate counts, not a stored history or completeness guarantee.

Saved model files, original catalog/clean data, all old ARE-007A reports and Notebook 01 had identical hashes before/after live execution. Notebook outputs/execution counts are cleared, and no profile data, tokens or raw API cache is part of the task's publishable files. Windows Jupyter emitted its usual event-loop/local TCP warnings but execution succeeded. No commit/push or ARE-008 work.
