"""Explicitly bounded, opt-in metadata research. Never called by inference."""

from datetime import datetime, timezone

import requests


QUERY = """query ($ids: [Int]) {
  Page(page:1,perPage:20) { media(idMal_in:$ids,type:ANIME) {
    idMal format relations { edges { relationType node { idMal type format } } }
  } }
}"""


def fetch_anilist_relations(mal_ids):
    if not mal_ids or len(mal_ids) > 20 or any(type(v) is not int or v <= 0 for v in mal_ids):
        raise ValueError("Explicitly request 1..20 positive MAL IDs; no crawling")
    try:
        response = requests.post("https://graphql.anilist.co", json={"query": QUERY,
            "variables": {"ids": sorted(set(mal_ids))}}, timeout=(10, 30))
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("Relations API unavailable; no retry or partial snapshot") from None
    if result.get("errors") or not isinstance(result.get("data", {}).get("Page", {}).get("media"), list):
        raise ValueError("Invalid relations API response")
    media = result["data"]["Page"]["media"]
    if len(media) > len(set(mal_ids)) or any(m.get("idMal") not in mal_ids for m in media):
        raise ValueError("Unexpected relations query result")
    return media


def relation_snapshot(media, requested_ids):
    """Keep only factual anime IDs, formats and relation types, with provenance."""
    nodes, edges = {}, set()
    for item in media:
        mal = item["idMal"]
        nodes[mal] = {"mal_id": mal, "format": item.get("format"), "queried": True}
        for edge in item["relations"]["edges"]:
            node = edge["node"]
            other = node.get("idMal")
            if node.get("type") != "ANIME" or type(other) is not int or other <= 0:
                continue
            nodes.setdefault(other, {"mal_id": other, "format": node.get("format"), "queried": False})
            edges.add((mal, other, edge["relationType"]))
    return {"schema_version": 1, "source": "AniList API v2 structured media relations",
        "source_url": "https://graphql.anilist.co", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "queried_mal_ids": sorted(set(requested_ids)), "nodes": [nodes[k] for k in sorted(nodes)],
        "relations": [{"source_mal_id": a, "target_mal_id": b, "relation": kind} for a, b, kind in sorted(edges)]}


def fetch_yummy_viewing_orders(client, anime_refs):
    """Explicit offline research: at most 3 details and 2 batches of 20 IDs.

    No HTML, collection owners, profiles, recursive discovery or partial writes.
    """
    if not anime_refs or len(anime_refs) > 3:
        raise ValueError("Explicitly select 1..3 anime detail pages")
    examples, ids = [], set()
    for ref in anime_refs:
        detail = client.get_anime(ref)
        order = detail.get("viewing_order")
        if not isinstance(order, list):
            raise ValueError("Detail response has no structured viewing_order")
        clean = []
        for row in order:
            anime_id = row.get("anime_id")
            if type(anime_id) is not int or anime_id <= 0:
                raise ValueError("Invalid source anime ID")
            ids.add(anime_id)
            clean.append({key: row.get(key) for key in ("anime_id", "type", "data")})
        examples.append({"seed_yummy_id": detail["anime_id"], "viewing_order": clean})
    if len(ids) > 40:
        raise ValueError("Explicit research scope exceeds 40 entries; no automatic expansion")
    resolved = []
    ordered_ids = sorted(ids)
    for start in range(0, len(ordered_ids), 20):
        batch = ordered_ids[start:start + 20]
        for row in client.get_anime_by_ids(batch):
            if row["anime_id"] not in batch:
                raise ValueError("Unexpected anime ID in resolution response")
            resolved.append({"anime_id": row["anime_id"],
                "myanimelist_id": (row.get("remote_ids") or {}).get("myanimelist_id")})
    return examples, resolved
