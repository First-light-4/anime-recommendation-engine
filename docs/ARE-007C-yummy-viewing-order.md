# ARE-007C addition: YummyAnime structured viewing order

Research and implementation complete locally; awaiting code review. No commit/push,
ALS training, dataset changes, HTML parsing, bulk catalog collection or ARE-008 work.

## Confirmed API source

The earlier investigation missed `viewing_order` in the detail response. That
conclusion is corrected here; the source is available through the existing public
X-Application API, without owner Bearer authorization.

The [official Swagger](https://ru.yummyani.me/api/swagger) documents:

- `GET /anime/{url}`: ID or alias; returns `viewing_order` with `anime_id`, `type`,
  `anime_url` and `data: {id,index,text}`. `data.id` is the shared source order-group
  identifier in the observed examples, `index` is the authored position, and `text`
  labels the role (continuation, addition, compilation, etc.).
- `GET /anime/{id}/collections`: user-owned public collections, with owners, likes
  and mixed anime lists. No `viewing_order` or reliable mandatory progression was
  established there. Collection owners or contents are not stored or used as a
  franchise/watch-order source.
- `GET /anime?ids=...`: resolves the listed YummyAnime IDs to `remote_ids.myanimelist_id`.
  The order items themselves do not provide MAL IDs in the examined responses.

Real checks returned HTTP 200 for the selected detail, collection and bounded
catalog lookups. Correct roots: Attack on Titan Yummy ID 45 / MAL 16498; Demon Slayer
595 / 38000; Cyberpunk: Edgerunners 4887 / 42310. Cyberpunk 2 detail ID 21748 returned
the same two-item order as the first part. Its MAL ID is 61990.

An initial Cyberpunk probe used the wrong MAL ID (44511, Chainsaw Man). That extra
detail/collection probe was discarded; none of its entries are in the final mapping.
Only three requested franchises were retained. All 32 distinct source entries were
resolved to MAL IDs in two ID-filtered batches (20 + 12). No recursive discovery or
per-recommendation relation fetching is used.

The array is an authored viewing order, not a guarantee of strictly ascending release
years: in the examined Attack on Titan order a 2014 OVA appears after a 2015 recap.
The implementation sorts the explicit `data.index`, never guessed titles or years.

## Mapping construction and storage

`ingestion/anime_relations.py::fetch_yummy_viewing_orders` is an explicit offline
helper: at most three detail pages and forty listed IDs, resolved in two batches.
It does not collect profiles, collection owners, HTML or arbitrary neighbor graphs.
`preprocessing/yummy_franchise_mapping.py::build_yummy_snapshot` normalizes the
responses and returns the versioned snapshot, without HTTP or file writes.

The active mapping is `configs/yummyanime_franchise_relations.json`, selected by
`configs/franchise_filter.json`. It records source URL/name, UTC retrieval timestamp,
normalization version, source group/index/role records, MAL resolution, generated
sequence stages and snapshot fingerprint through the existing loader. Raw public
research metadata remains only under ignored `data/external/`; no profiles are in it.

The API's exact role labels identify the first adaptation/original entry and
`продолжение` stages. Labels for additions, side stories and compilations do not
create required seasons. Unknown labels mark the order unsafe: retain known family
membership, publish unknown-order diagnostics, and do not invent a mandatory chain.
Duplicate indices, gaps, conflicting group IDs and contradictory repeated orders
are rejected. An unsorted response is ordered by its unique source indices.

Season chains are generated from those source records rather than manually listed.
The older AniList snapshot is retained solely as separate historical evidence and
for its already reviewed Mugen Train movie/TV ALTERNATIVE pair. A generic alternative
or compilation label does not prove full completion equivalence: the Attack on Titan
recap movie is deliberately NOT substituted for completion of the whole first TV
season. This conservative supplemental exception and its fingerprint are recorded.

An unresolved MAL ID is an explicit empty stage, not a deleted element. The filter
cannot bridge that gap to recommend a later part. Frontier parts absent from the
training data remain in the source sequence, without fictitious model factors.

## Updated recommendation policy

- Exact observed parts are excluded regardless of rating/status.
- Only completed proves viewing; planned/watching do not advance progression.
- Use the most advanced completed mainline stage as evidence to withhold earlier
  stages even if they were not manually listed. Do not mark the full franchise seen.
- With no completed stage, the first stage is the eligible entry. Otherwise only
  the next stage is eligible. Confirmed movie/TV alternative IDs belong to one stage,
  not two mandatory consecutive stages; completing either advances that stage.
- A missing next stage in train vocabulary or MAL mapping never permits jumping
  to a later stage. If no admissible candidates remain, return fewer results after
  the bounded local pool is exhausted, rather than insert prohibited seasons.
- Optional films/spin-offs are never counted as mandatory seasons or automatically
  seen. When a true next stage is available and not already observed, that stage is
  prioritized over optional entries via an explicit `optional_deferred` reason.
  Optional parts remain eligible after progression finishes or when no such next
  stage can be offered; they are not treated as substitute later seasons.
- Keep one item per known franchise by default. Unknown families are not inferred
  from similar names. Scores and relative order of eligible candidates are unchanged.

Post-processing remains a separate local stage, used identically for ALS and the
existing Popularity fallback. Pool expansion is local, up to train vocabulary.
Online inference does not query detail pages or relations; only the existing
bounded profile/title requests can occur. Mapping refresh is an explicit offline
operation, not an automatic expansion of source coverage.

## Case results

These are deterministic post-processing checks against the real static source
mapping and controlled candidate rankings, not a new training run or quality claim.

| Franchise | Completed evidence | Eligible mainline stage |
| --- | --- | --- |
| Attack on Titan | none | MAL 16498, first season |
| Attack on Titan | first season | 25777, season 2 |
| Attack on Titan | first and second | 35760, season 3 |
| Attack on Titan | late part 38524, earlier parts unmarked | 40028; earlier parts withheld |
| Demon Slayer | none | 38000, first season |
| Demon Slayer | first season | one of 40456 / 49926, Mugen Train alternatives |
| Demon Slayer | Mugen Train movie OR TV completed | 47778, Entertainment District |
| Cyberpunk | none | 42310, first part |
| Cyberpunk | first part completed | next is 61990, but absent from train: no later substitute |

Tests additionally cover planned/watching, multiple high-scoring seasons, missing
intermediate MAL IDs, next-stage model absence, optional movie/spin-off handling,
unknown role labels, source conflicts, cap=1, stable scores and no HTTP in filtering.

## Coverage and evaluation

32/32 order entries resolve to MAL; 28 are present in the full local catalog and
22 in the train vocabulary. The active snapshot covers three franchises and
22/2091 train items (1.052%), versus 20/2091 (0.956%) with the initial snapshot.
The additional covered train IDs are 39477 and 42310; none of the previously covered
train items are lost. This is a source-quality improvement and a small targeted
coverage increase, not a full-catalog solution. MAL 61990 (Cyberpunk 2) is known to
the mapping but has no trained item factor.

A separate `ARE-007C-yummy-order-v2` validation report was generated under ignored
`artifacts/evaluation/are007c/`, leaving ARE-007A and prior ARE-007C reports unchanged.
Full Recall@10 is 0.252278 and NDCG@10 0.353445 after filtering, versus original ALS
0.283710 / 0.388166. Catalog coverage is 236/2091 (11.2865%) vs 233/2091 (11.1430%).
104/1000 original entries are replaced. Aggregate Recall/NDCG match the initial
partial-filter run here, although the selected items and protocol fingerprint differ.
The ratings-only validation has no completed proof, so completed is not imputed from
ratings. This remains a conservative diagnostic rather than full online evaluation.

## Verification

162 tests passed in 6.52 seconds. Notebook 05 executed top to bottom with a synthetic
completed profile, HTTP and ALS fit disabled, producing Top-15 through the new local
mapping. No additional real user-profile requests were needed. Saved model/data
files, old ARE-007A artifacts and Notebook 01 had unchanged hashes. Notebook outputs
and execution counts remain empty; no tokens or user lists are stored. No ALS
retraining, ARE-008, commit or push.
