"""Build a versioned local mapping from structured YummyAnime viewing_order.

No title similarity, release-year sorting, per-item HTTP or implicit movie rules.
"""

from datetime import datetime, timezone
import hashlib
import json


ENTRY_LABELS = {"адаптация манги", "адаптация игры", "адаптация романа", "адаптация ранобэ", "оригинальный сюжет", "оригинальная история"}
OPTIONAL_LABELS = {"дополнение", "ответвление сюжета", "спин-офф", "альтернативная версия",
    "компиляция", "компиляция сериала", "компиляция сериалов", "компиляция спешла", "компиляция фильма"}


def build_yummy_snapshot(examples, resolved_rows, alternative_snapshot=None):
    lookup = {}
    for row in resolved_rows:
        yummy, mal = row["anime_id"], row.get("myanimelist_id")
        mal = mal if type(mal) is int and mal > 0 else None
        if yummy in lookup and lookup[yummy] != mal:
            raise ValueError("Ambiguous YummyAnime to MAL mapping")
        lookup[yummy] = mal
    nodes, families, edges, sequences, mainline, entries, source_orders = {}, {}, set(), [], set(), set(), []
    group_signatures = {}
    for example in examples:
        rows = example["viewing_order"]
        if not rows:
            continue
        source_ids = {r["data"]["id"] for r in rows}
        indices = [r["data"]["index"] for r in rows]
        if len(source_ids) != 1 or any(type(i) is not int or i < 0 for i in indices) or sorted(indices) != list(range(len(rows))):
            raise ValueError("Inconsistent, duplicate or incomplete source order indices")
        if any(type(i) is not int or i <= 0 for i in source_ids) or len({r["anime_id"] for r in rows}) != len(rows):
            raise ValueError("Invalid viewing_order IDs")
        ordered = sorted(rows, key=lambda r: r["data"]["index"])
        family = "yummy-family-" + str(next(iter(source_ids)))
        signature = [(r["anime_id"], r["data"]["index"], r["data"].get("text")) for r in ordered]
        if family in group_signatures:
            if group_signatures[family] != signature:
                raise ValueError("Conflicting viewing_order for the same source family")
            continue
        group_signatures[family] = signature
        steps, optional, safe, source_rows = [], [], True, []
        for row in ordered:
            mal = lookup.get(row["anime_id"])
            label = str(row["data"].get("text", "")).strip().casefold()
            idx = row["data"]["index"]
            role = "entry" if idx == 0 and label in ENTRY_LABELS else ("continuation" if label == "продолжение" else ("optional" if label in OPTIONAL_LABELS else "unknown"))
            safe &= role != "unknown" and (idx != 0 or role == "entry")
            if mal is not None:
                if mal in families and families[mal] != family:
                    raise ValueError("MAL item belongs to conflicting source families")
                families[mal] = family
                nodes[mal] = {"mal_id": mal, "format": row.get("type", {}).get("alias"), "queried": True}
            if role in {"entry", "continuation"}:
                # An empty MAL list is a barrier, never silently remove/bridge it.
                steps.append({"mal_ids": [mal] if mal else [], "yummy_ids": [row["anime_id"]], "source_index": idx})
            elif mal:
                optional.append(mal)
            source_rows.append({"yummy_id": row["anime_id"], "mal_id": mal, "index": idx, "label": row["data"].get("text"), "role": role})
        family_mals = sorted(m for m, f in families.items() if f == family)
        for mal in family_mals[1:]:
            edges.add((family_mals[0], mal, "YUMMY_FAMILY"))
        # A separately verified ALTERNATIVE edge may substitute completion evidence.
        # A free-text compilation label alone does not identify its source version.
        if safe and alternative_snapshot:
            reviewed_alternatives = set(alternative_snapshot["reviewed_policy"]["mainline_mal_ids"])
            for edge in alternative_snapshot["relations"]:
                a, b = edge["source_mal_id"], edge["target_mal_id"]
                if edge["relation"] != "ALTERNATIVE" or not {a, b} <= reviewed_alternatives or families.get(a) != family or families.get(b) != family:
                    continue
                for step in steps:
                    if (a in step["mal_ids"]) != (b in step["mal_ids"]):
                        alternate = b if a in step["mal_ids"] else a
                        if alternate in optional:
                            step["mal_ids"].append(alternate)
                            step["mal_ids"].sort()
                            optional.remove(alternate)
                            edges.add((a, b, "ALTERNATIVE"))
        if safe:
            for step in steps:
                mainline.update(step["mal_ids"])
            entries.update(steps[0]["mal_ids"])
            for previous, current in zip(steps, steps[1:]):
                for a in previous["mal_ids"]:
                    for b in current["mal_ids"]:
                        edges.add((a, b, "SEQUEL"))
        sequences.append({"franchise_id": family, "order_known": safe, "steps": steps, "optional_mal_ids": optional})
        source_orders.append({"franchise_id": family, "seed_yummy_id": example["seed_yummy_id"], "items": source_rows})
    return {"schema_version": 1, "source": "YummyAnime GET /anime/{url}.viewing_order",
        "source_url": "https://ru.yummyani.me/api/swagger", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "normalization_version": "yummy-viewing-order-v1", "nodes": [nodes[m] for m in sorted(nodes)],
        "relations": [{"source_mal_id": a, "target_mal_id": b, "relation": kind} for a, b, kind in sorted(edges)],
        "family_ids": {str(m): f for m, f in sorted(families.items())},
        "ordered_sequences": sequences, "source_orders": source_orders,
        "supplemental_alternative_source": alternative_snapshot.get("source") if alternative_snapshot else None,
        "supplemental_alternative_fingerprint": hashlib.sha256(json.dumps(alternative_snapshot, sort_keys=True).encode()).hexdigest() if alternative_snapshot else None,
        "reviewed_policy": {"mainline_mal_ids": sorted(mainline), "entry_mal_ids": sorted(entries),
            "scope": "Generated from source index and exact data.text role labels; optional/unknown roles are not mandatory seasons"}}
