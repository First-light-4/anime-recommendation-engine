# ARE-007C.2 — automatic viewing-order metadata pipeline

Code and synthetic integration tests are ready for review. The live Rascal acceptance
check remains blocked by HTTP 502 from the catalog lookup. No full-catalog scan,
model training, commit/push or ARE-008 work was performed.

## Files and responsibilities

```text
configs/viewing_order_metadata.json              budget/rate/TTL/path configuration
configs/franchise_filter.json                   prefer prepared local artifact directory
src/anime_recommender/
  ingestion/metadata_requests.py                attempt budget, rate limiting, backoff
  ingestion/viewing_order_metadata.py           discovery, checkpoint/resume, publishing
  ingestion/franchise_artifacts.py              verified local loading and candidate coverage
  ingestion/yummyanime.py                       Retry-After exposed through sanitized errors
  preprocessing/metadata_orders.py             merge overlap/precedence/quality checks
  preprocessing/franchise_mapping.py          prepared/static/empty fallback loading
  inference/franchise_filter.py               extra coverage/exclusion diagnostics
  recommendation_exploration.py              metadata warning/source in existing inference
notebooks/05_yummyanime_recommendations.ipynb
tests/test_viewing_order_metadata_pipeline.py
README.md
docs/ARE-007C.2-metadata-pipeline.md
```

The existing transport, source normalization, structured role policy, confirmed
alternative-stage support and fixed-model franchise post-processing are reused.
Notebook 01's existing local user edit is untouched.

## Scope and execution

The default source is every MAL ID in the current train item mapping (2091 items),
not manually selected title strings. A diagnostic MAL subset is an optional CLI
parameter; no named franchise/season IDs are hardcoded in the discovery algorithm.
Existing metadata can bootstrap the cache without limiting the discoverable catalog.

```powershell
# Default: dry-run; no network, token requirement, checkpoint or artifact writes.
.venv\Scripts\python.exe -m anime_recommender.ingestion.viewing_order_metadata

# Explicitly bounded diagnostic/resume; do not loop while the provider returns 502.
.venv\Scripts\python.exe -m anime_recommender.ingestion.viewing_order_metadata --execute --max-requests 5 --mal-ids 37450,38329
```

The all-vocabulary dry-run saw 2091 train items and 2069 pending MAL lookups after
bootstrapping the existing cache; batch_size=10 gives 207 pending batches. It made
zero requests and wrote no artifacts. Discovery budgets above 10 attempts are rejected
unless an explicit large-run flag is supplied after separate authorization. That
flag has not been used. There is no scheduler or background catalog loop.

Discovery resolves MAL IDs in bounded catalog batches, fetches the detail viewing_order
once for a known group, and deduplicates sibling detail requests by source membership.
It resolves all listed external Yummy IDs in batches, retaining parts not present in
the ALS vocabulary. Pending work is derived from checkpoint caches on resume. No HTML,
title similarity, user-profile endpoint or collection owners are used.

## Request and recovery policy

- Default budget: 5 HTTP attempts per run, including every retry.
- Default rate: 0.5 requests/second; this is a conservative configured setting,
  not a claimed official provider limit.
- Two retries with exponential backoff, only for network/timeout errors and selected
  temporary HTTP statuses (408,425,429,500,502,503,504). Invalid JSON/schema and other
  permanent failures are not retried within the operation.
- Retry-After supports seconds and HTTP dates. If the wait exceeds the configured
  max_sleep_seconds=30, checkpoint and defer to a later run rather than retry early.
- 401/403 stops acquisition immediately. No authorization bypass or private token.
- Cache TTL is 30 days, negative/ambiguous entries one day. Refresh does not redownload
  fresh mappings or known groups. Newly added vocabulary members can trigger discovery.
- Checkpoint after every attempt/result. A killed process can replay one in-flight
  request whose result was not committed; already persisted successful work is reused.
- Temporary refresh failure retains previously known source data with stale/error
  warnings. A complete fresh page can supersede stale overlapping source pages; fresh
  contradictory pages instead remain a reported conflict.

Use one acquisition process at a time: this implementation is not a multi-worker
distributed crawler. Checkpoints and version-pointer writes are atomic, but concurrent
writers are outside the supported operational protocol. Read-only inference can run
against the previously published version while acquisition publishes the next one.

## Merge, interpretation and confidence

Groups merge through shared Yummy IDs or source group IDs, never names. The merged
precedence graph must have a unique topological order. Cycles, conflicting roles,
duplicate indices, source gaps or ambiguous branches retain family membership but
disable mandatory chronology. Duplicate/conflicting MAL identities are quarantined,
not arbitrarily selected. Detail identity must match its requested Yummy ID.

Source role labels distinguish mainline, optional and alternative entries; type or
release order alone does not imply mandatory viewing. A generic alternative caption
does not identify an exact equivalent prerequisite stage. Such entries stay optional
unless a separately verified structured alternative edge is available. The existing
reviewed Mugen Train movie/TV edge is an optional supplemental cache, not a required
manual franchise list or an automatic rule for every compilation/remake. Synthetic
tests verify a confirmed alternate completes the same stage. Unknown interpretations
are published explicitly, not forced into a linear sequence.

Missing MAL IDs remain empty required stages; next-stage absence from the model never
permits a later season. Existing completed/planned/watching, exact seen filtering,
late-completed ancestry, max-one-family and local pool expansion rules are unchanged.

## Local artifacts

```text
data/processed/franchises/
  state.json
  latest.json
  versions/<UTC-run>/
    mapping.json
    parts.parquet
    source_records.json
    coverage.json
    manifest.json
```

All files are ignored by Git. Parts record Yummy ID, nullable MAL ID, generated local
family ID, original source group IDs, viewing index, format, mainline/alternative/
optional/unknown role, original caption, stage group, source, actual source retrieval
time, build time, schema version, completeness and order-confidence status. Public
source records are versioned so a later cache refresh does not destroy old provenance.
No response user fields, personal lists, owner records or token values are stored.
The two older local metadata snapshots under configs/ are now explicitly ignored
too; operational configuration remains source code. Clean checkouts do not need
those snapshots for synthetic tests. Two optional real-source smoke tests skip if
local source metadata is unavailable; in the reported local run both executed.

Each immutable version has SHA-256 checksums; latest.json is atomically moved only
after all version files are complete. The loader validates paths, schema and hashes.
Mapping fingerprints remain identical on a zero-network repeat with unchanged cached
facts. Build timestamps/output versions can differ. Negative IDs, error classes,
unresolved mappings, stale orders and conflicts are reported separately.

Notebook 05 loads verified prepared data first, then a static partial snapshot, then
an empty relationship mapping with exact seen filtering if neither exists. Missing or
damaged artifacts never trigger online discovery. Warnings distinguish partial/stale/
failed refreshes. Diagnostics show known groups, covered candidate share, duplicate
season exclusions, unsuitable continuation exclusions and final count.

## Actual small live run

Only MAL 37450 and 38329 were selected. Two bounded runs were attempted, with a pause
and checkpoint resume between them. Both catalog lookups returned HTTP 502 on all
three configured attempts. Further real requests were stopped.

| Observation | Result |
| --- | --- |
| Selected MAL IDs attempted | 2 |
| Successfully resolved new selected IDs | 0 |
| HTTP attempts | 3 + 3 = 6 |
| HTTP failures | 6 × 502 |
| Failed batch operations | 2 |
| Newly discovered groups/parts | 0 / 0 |
| Preserved cached groups/parts | 3 / 32 |
| Ambiguous identifiers/conflicting cached groups | 0 / 0 |
| Covered train vocabulary | 22 / 2091 = 1.0521% |
| Validation original ALS Top-15 candidate occurrences covered | 194 / 1500 = 12.9333% |
| Unique items in those Top-15 lists | 298 |

Candidate coverage counts occurrences with known family membership before franchise
filtering. It is not ranking quality or full chronology coverage. That measurement
used only the existing local validation reader/saved models, with no test data or fit.
All existing coverage came from the bootstrap cache, not new successful acquisition.

The live Rascal API sequence therefore is NOT confirmed and is not inserted manually.
The real model can still treat those items as unknown until source acquisition succeeds;
the task's live pair guarantee remains unverified. An explicitly synthetic API regression
using those MAL IDs verifies that once a confirmed group is acquired, a new-user Top-15
contains only the first stage, not both. This is not represented as live evidence.

## Full-run estimate and API conditions

With batch_size=10, a cold seed lookup costs up to 210 requests. Detail discovery is
bounded by the number of distinct uncovered source groups/standalone items, conservatively
up to 2091 seed details before deduplication. Resolving external members and retries adds
requests. A planning formula is `lookup_batches + distinct_needed_details + external_ID_batches + retry_attempts`.
The seeds-only conservative base is 2301 requests, about 77 minutes at
0.5 requests/second, excluding latency, external resolution and retries. It is not an
upper bound for the whole catalog; no empirical discovery/dedup rate can be inferred
from the failed live Rascal attempts. Budgets and checkpoints allow approved work to
be split into runs without an uncontrolled catalog crawl.

The public Swagger documents the required operations but did not establish a published
bulk-metadata collection allowance or official rate limit. Absence of such documentation
is not treated as permission. A full scan still requires separate user approval and
confirmation of appropriate usage terms with YummyAnime. No full scan was performed.

## Verification and remaining limitation

177 tests passed in 7.28 seconds: mock acquisition, MAL conflicts, overlap merging,
request deduplication, dry-run, rate limiting, 429/Retry-After/date parsing, backoff,
authorization stop, budgets, interruption/resume, TTL refresh/negative cache, stale
source preservation/supersession, alternative stages, source gaps, artifact corruption,
missing-data fallback and the synthetic Rascal Top-15 case. Existing recommendation,
completion, late-history, next-missing, score preservation and no-network filter tests pass.

Notebook 05 completed top to bottom using prepared local artifacts and a synthetic
profile, with HTTP and ALS fit patched to fail. Top-15 and metadata warnings were
displayed. Original model/data files, ARE-007A reports and Notebook 01 hashes were
unchanged. No notebook outputs/counts or private user data were saved. Live Rascal
verification remains pending API recovery; no further retries, full discovery,
commit/push or ARE-008 work was started.

## Authorized pre-publication review

Final local tests: 177 passed. A separate clean-checkout simulation with local
metadata hidden passed 175 tests and skipped only the two optional real-source
smoke tests. No files were deleted or changed for that simulation.

The user authorized publication of completed ARE-007C/C.2 code to personal main.
Three single-attempt API diagnostics were made, with no automatic retries:

- `GET /anime?mal_ids=37450,38329&limit=20&offset=0`: HTTP 502.
- `GET /anime/45` (known YummyAnime ID, previously successful): HTTP 502.
- `GET /anime?mal_ids=37450&limit=20&offset=0`: HTTP 502.

The current Swagger JS returned HTTP 200 and still declares server api.yani.tv,
integer-array mal_ids, detail aliases OR YummyAnime IDs, and viewing_order in the
detail response. The catalog parameter contains MAL IDs; 45 is used only as a
YummyAnime detail ID. No documented contract/base-URL change was found. Because
the known detail also fails, the observed 502 is not isolated to the pair lookup
or its comma-separated encoding. The underlying external gateway/backend cause
is not established; no further YummyAnime requests were attempted.

Actual checkpoint entries for both Rascal MAL IDs were verified as status=error,
retryable=True, not missing. A new regression fails a lookup with 502, restores
the mock API and resumes the same checkpoint successfully before the negative TTL
expires. Request/backoff, group deduplication, local-only loading and all tests
pass. Live Rascal acquisition remains pending provider recovery; it is not claimed
as complete or replaced by invented metadata. Notebook 01 and every metadata/data
artifact are excluded from the planned commit.
