"""Structured franchise components and explicitly reviewed mainline partial order."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


FAMILY_RELATIONS = {"PREQUEL", "SEQUEL", "PARENT", "SIDE_STORY", "SPIN_OFF", "ALTERNATIVE", "SUMMARY", "COMPILATION", "YUMMY_FAMILY"}


@dataclass
class FranchiseMapping:
    franchise: dict
    groups: dict
    parents: dict
    mainline: set
    entry_groups: set
    snapshot: dict
    fingerprint: str

    def sequence_state(self, mal_id, completed_mal_ids):
        """Next required source stage; an unmapped stage is preserved as a barrier."""
        family = self.franchise.get(mal_id)
        sequence = next((s for s in self.snapshot.get("ordered_sequences", []) if s["franchise_id"] == family and s["order_known"]), None)
        if sequence is None:
            return None
        completed = set(completed_mal_ids)
        steps = sequence["steps"]
        last = max((i for i, s in enumerate(steps) if set(s["mal_ids"]) & completed), default=-1)
        candidate = next((i for i, s in enumerate(steps) if mal_id in s["mal_ids"]), None)
        next_index = last + 1
        return {"candidate_step": candidate, "last_completed_step": last,
            "next_step": next_index, "next_mal_ids": steps[next_index]["mal_ids"] if next_index < len(steps) else [],
            "finished": next_index == len(steps)}

    def ancestors(self, mal_id):
        """None means unknown/ambiguous order; empty tuple means confirmed entry."""
        if mal_id not in self.mainline:
            return None
        current = self.groups[mal_id]
        ancestors, visited = [], set()
        while current not in self.entry_groups:
            if current in visited:
                raise ValueError("Cyclic mainline relation")
            visited.add(current)
            parents = self.parents.get(current, set())
            if len(parents) != 1:
                return None
            current = next(iter(parents))
            ancestors.append(current)
        return tuple(ancestors)

    def describe(self, mal_ids):
        import pandas as pd
        rows = []
        formats = {row["mal_id"]: row.get("format") for row in self.snapshot["nodes"]}
        for mal in sorted(set(mal_ids)):
            order = self.ancestors(mal)
            rows.append({"myanimelist_id": mal, "franchise_id": self.franchise.get(mal),
                "format": formats.get(mal), "reviewed_mainline": mal in self.mainline,
                "order_known": order is not None, "ancestor_groups": len(order) if order is not None else None})
        return pd.DataFrame(rows)

    def viewing_path(self, mal_id):
        """Ordered prerequisite groups, not a total order for an entire franchise.

        Multiple IDs within one tuple are explicitly reviewed alternatives.
        None means no trustworthy required watch path for this item.
        """
        ancestors = self.ancestors(mal_id)
        if ancestors is None:
            return None
        group_order = [*reversed(ancestors), self.groups[mal_id]]
        return tuple(tuple(sorted(v for v in self.mainline if self.groups[v] == group)) for group in group_order)


def build_franchise_mapping(snapshot):
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unknown relation snapshot schema")
    nodes = [row["mal_id"] for row in snapshot["nodes"]]
    if len(nodes) != len(set(nodes)) or any(type(v) is not int or v <= 0 for v in nodes):
        raise ValueError("Invalid relation node IDs")
    policy = snapshot["reviewed_policy"]
    mainline, entries = set(policy["mainline_mal_ids"]), set(policy["entry_mal_ids"])
    if not entries <= mainline <= set(nodes):
        raise ValueError("Mainline/entry policy references unknown IDs")
    connected, equivalent = {v: v for v in nodes}, {v: v for v in nodes}
    def root(graph, value):
        while graph[value] != value:
            value = graph[value]
        return value
    def union(graph, a, b):
        a, b = root(graph, a), root(graph, b)
        graph[max(a, b)] = min(a, b)
    for edge in snapshot["relations"]:
        a, b, kind = edge["source_mal_id"], edge["target_mal_id"], edge["relation"]
        if a not in connected or b not in connected or a == b:
            raise ValueError("Invalid relation endpoints")
        if kind in FAMILY_RELATIONS:
            union(connected, a, b)
        # Alternatives only substitute prerequisite evidence within reviewed mainline;
        # they do not imply that the alternate item itself was watched.
        if kind == "ALTERNATIVE" and a in mainline and b in mainline:
            union(equivalent, a, b)
    franchise = {v: "mal-family-" + str(root(connected, v)) for v in nodes
                 if any(v in (e["source_mal_id"], e["target_mal_id"]) and e["relation"] in FAMILY_RELATIONS for e in snapshot["relations"])}
    groups = {v: root(equivalent, v) for v in nodes}
    if "family_ids" in snapshot:
        declared = {int(v): family for v, family in snapshot["family_ids"].items()}
        if set(declared) != set(nodes):
            raise ValueError("Invalid source family map")
        franchise = declared
    parents = {}
    for edge in snapshot["relations"]:
        a, b, kind = edge["source_mal_id"], edge["target_mal_id"], edge["relation"]
        if kind == "PREQUEL":
            a, b = b, a
        if kind not in {"PREQUEL", "SEQUEL"} or a not in mainline or b not in mainline:
            continue
        if groups[a] != groups[b]:
            parents.setdefault(groups[b], set()).add(groups[a])
    result = FranchiseMapping(franchise, groups, parents, mainline, {groups[v] for v in entries}, snapshot,
        hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest())
    # Reject cycles, including a false entry declaration with an incoming mainline edge.
    if any(parents.get(group) for group in result.entry_groups):
        raise ValueError("Confirmed entry has a mainline predecessor")
    def check_cycle(group, active, finished):
        if group in active:
            raise ValueError("Cyclic mainline relation")
        if group in finished:
            return
        for parent in parents.get(group, set()):
            check_cycle(parent, active | {group}, finished)
        finished.add(group)
    finished = set()
    for group in set(groups.values()):
        check_cycle(group, set(), finished)
    return result


def load_franchise_resources(root):
    root = Path(root).resolve()
    config = json.loads((root / "configs/franchise_filter.json").read_text(encoding="utf-8"))
    required = {"enabled", "relations_path", "max_recommendations_per_franchise", "candidate_multiplier"}
    if not required <= set(config) <= required | {"prepared_dir"} or type(config["enabled"]) is not bool:
        raise ValueError("Invalid franchise filter config")
    for key in ("max_recommendations_per_franchise", "candidate_multiplier"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Franchise limits must be positive integers")
    warning = None
    if config.get("prepared_dir"):
        directory = (root / config["prepared_dir"]).resolve()
        if not directory.is_relative_to(root / "data/processed/franchises"):
            raise ValueError("Prepared metadata must be under data/processed/franchises")
        try:
            from anime_recommender.ingestion.franchise_artifacts import load_prepared_mapping
            snapshot, coverage = load_prepared_mapping(directory)
            config["metadata_warning"] = "Prepared metadata has partial coverage; unknown relations remain unfiltered"
            config["metadata_source"] = "prepared local version"
            if coverage.get("stale_orders"):
                config["metadata_warning"] += "; stale source orders are present"
            if coverage.get("errors_this_run") or coverage.get("failed_operations_this_run") or coverage.get("unresolved_seed_errors"):
                config["metadata_warning"] += "; last metadata update had API errors"
            if coverage.get("unknown_order_groups"):
                config["metadata_warning"] += "; some group orders are unresolved"
            return config, build_franchise_mapping(snapshot)
        except (OSError, ValueError, KeyError):
            warning = "Prepared metadata missing/invalid; using available static partial mapping"
    path = root / config["relations_path"]
    if not path.resolve().is_relative_to((root / "configs").resolve()):
        raise ValueError("Relation snapshot must be under configs/")
    try:
        mapping = build_franchise_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        mapping = build_franchise_mapping({"schema_version": 1, "nodes": [], "relations": [],
            "reviewed_policy": {"mainline_mal_ids": [], "entry_mal_ids": []}})
        warning = "No valid local franchise metadata; exact seen filtering only, franchise coverage is unknown"
    config["metadata_warning"] = warning or "Static metadata has partial coverage"
    config["metadata_source"] = "static fallback"
    return config, mapping
