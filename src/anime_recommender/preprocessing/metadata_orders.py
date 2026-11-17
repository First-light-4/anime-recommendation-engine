"""Merge overlapping source orders without guessing title/branch equivalence."""

import copy

from anime_recommender.preprocessing.yummy_franchise_mapping import build_yummy_snapshot


def merge_metadata_orders(examples, resolved, supplement=None):
    """Unique topological order only; contradictions preserve family, disable order."""
    examples = sorted(examples, key=lambda e: e["seed_yummy_id"])
    parent = list(range(len(examples)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    by_anime, by_group = {}, {}
    for i, example in enumerate(examples):
        for row in example["viewing_order"]:
            for table, key in ((by_anime, row["anime_id"]), (by_group, row["data"]["id"])):
                if key in table:
                    parent[root(i)] = root(table[key])
                table[key] = i
    components = {}
    for i, example in enumerate(examples):
        components.setdefault(root(i), []).append(example)
    merged, quality = [], []
    mal_by_yummy = {row["anime_id"]: row.get("myanimelist_id") for row in resolved}
    for component in sorted(components.values(), key=lambda c: min(r["anime_id"] for e in c for r in e["viewing_order"])):
        rows, edges, roles, source_groups = {}, set(), {}, set()
        conflict, complete = False, True
        for example in component:
            order = example["viewing_order"]
            indices = [r["data"]["index"] for r in order]
            if len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
                conflict = True
            if sorted(indices) != list(range(len(indices))):
                complete = False
            ordered = sorted(order, key=lambda r: r["data"]["index"])
            for row in ordered:
                yummy = row["anime_id"]
                label = str(row["data"].get("text", "")).strip().casefold()
                if yummy in roles and roles[yummy] != label:
                    conflict = True
                rows[yummy] = copy.deepcopy(row)
                roles[yummy] = label
                source_groups.add(row["data"]["id"])
            edges.update((a["anime_id"], b["anime_id"]) for a, b in zip(ordered, ordered[1:]))
        incoming = {y: set() for y in rows}
        for a, b in edges:
            incoming[b].add(a)
        order, remaining, unique = [], set(rows), True
        while remaining:
            available = sorted(y for y in remaining if not (incoming[y] & remaining))
            if len(available) != 1:
                unique = False
            if not available:
                break
            chosen = available[0]
            order.append(chosen); remaining.remove(chosen)
        if remaining:
            conflict = True
            order += sorted(remaining)
        mals = [mal_by_yummy.get(y) for y in rows if mal_by_yummy.get(y)]
        if len(mals) != len(set(mals)):
            conflict = True  # Multiple Yummy IDs to one MAL are not invented alternatives.
        group_id = min(rows)
        synthetic = []
        for index, yummy in enumerate(order):
            row = rows[yummy]
            row["data"] = {"id": group_id, "index": index,
                "text": row["data"].get("text") if unique and not conflict and complete else "unknown/conflicting source order"}
            synthetic.append(row)
        merged.append({"seed_yummy_id": component[0]["seed_yummy_id"], "viewing_order": synthetic})
        quality.append({"franchise_id": "yummy-family-" + str(group_id), "source_group_ids": sorted(source_groups),
            "source_pages": len(component), "parts": len(rows), "unique_order": unique,
            "conflicting": conflict, "source_complete": complete,
            "status": "conflicting" if conflict else ("incomplete" if not complete else ("ordered" if unique else "ambiguous"))})
    snapshot = build_yummy_snapshot(merged, resolved, supplement)
    snapshot["normalization_version"] = "incremental-viewing-order-v2"
    snapshot["merge_quality"] = quality
    return snapshot
