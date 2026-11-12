# ADR-001 — Online YummyAnime positives must not require explicit ratings

Status: accepted architectural requirement. Date: 2026-11-12.
Implementation: deferred; this decision does not implement inference or train ALS.

## Decision

Offline training feedback and online profile feedback are separate policies.
The offline rating threshold defaults to >=5. Online personalization MUST NOT
depend exclusively on the presence of explicit scores: a user with many watched
anime and no scores must still receive personalized recommendations when their
history intersects the model's vocabulary.

Future online fold-in must support evidence from:

- Completed/watched anime without a score.
- Anime with a configured positive explicit score.
- Favorite anime, including favorites without scores.
- Optionally a weaker currently-watching signal, subject to validation.

Zero score alone is not positive evidence and is not a dislike. A separate watched
status or favorite flag can establish positive evidence even when score is zero
or missing. The exact semantics of YummyAnime zero scores remain to be checked
against the UI; do not infer watched status from the score.

Map the selected profile signals through MAL ID to the frozen train item mapping.
Only known model items enter fold-in. Preserve unknown-item counts in coverage.
Use popularity fallback when this process leaves NO usable positive interactions
with known model items, including an unknown-only history or empty profile.
Do not require five online positives: the offline train 5/5 filters are not an
online eligibility rule. Users with usable watched/favorite signals must not fall
back merely because they did not score their anime.

## Deferred details

Signal confidence weights, whether currently-watching is enabled, duplicate-signal
combination and conflicts between a low explicit rating and watched/favorite status
require a separately tested policy. Do not invent numerical weights in this stage.
No target data may be used to select a user's signals or calculate their factors.

## Consequences

Rating-only offline holdouts do not fully simulate unrated YummyAnime histories.
Future evaluation must state that domain gap and validate the additional signals.
Report personalization coverage, cold-item coverage and fallback usage alongside
ranking quality. ALS factors, API inference endpoints, recommendation serving and
fallback implementation remain outside ARE-005.1.
