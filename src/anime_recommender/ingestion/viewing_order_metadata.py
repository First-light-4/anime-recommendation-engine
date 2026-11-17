"""Budgeted resumable discovery over the whole train vocabulary. Dry-run by default."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import pandas as pd

from anime_recommender.ingestion.metadata_requests import MetadataRequests, RunStopped
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError, YummyAnimeNetworkError
from anime_recommender.preprocessing.franchise_mapping import build_franchise_mapping
from anime_recommender.preprocessing.metadata_orders import merge_metadata_orders


def _fresh(record, config, now):
    if not record or record.get("retryable"):
        return False
    days = config["negative_refresh_days"] if record.get("status") in {"missing", "empty", "ambiguous", "error"} else config["refresh_days"]
    return now - record.get("fetched_at", -1e12) < days * 86400


def _bootstrap(state, root, path):
    if state.get("bootstrapped") or not path or not (root / path).exists():
        return
    snapshot = json.loads((root / path).read_text(encoding="utf-8"))
    stamp = datetime.fromisoformat(snapshot["retrieved_utc"]).timestamp()
    formats = {node["mal_id"]: node.get("format") for node in snapshot["nodes"]}
    for order in snapshot.get("source_orders", []):
        rows = []
        group = int(order["franchise_id"].rsplit("-", 1)[-1])
        for row in order["items"]:
            rows.append({"anime_id": row["yummy_id"], "type": {"alias": formats.get(row.get("mal_id"))},
                "data": {"id": group, "index": row["index"], "text": row["label"]}})
            if row.get("mal_id"):
                mal, yummy = row["mal_id"], row["yummy_id"]
                state["mal"][str(mal)] = {"yummy_id": yummy, "status": "mapped", "fetched_at": stamp}
                state["yummy"][str(yummy)] = {"mal_id": mal, "status": "mapped", "fetched_at": stamp}
        state["orders"][str(order["seed_yummy_id"])] = {"seed_yummy_id": order["seed_yummy_id"],
            "viewing_order": rows, "status": "ordered", "fetched_at": stamp}
    state["bootstrapped"] = True


def _record_resolution(state, yummy, mal, now):
    mal = mal if type(mal) is int and mal > 0 else None
    previous = state["yummy"].get(str(yummy))
    if previous and previous.get("mal_id") and mal and previous["mal_id"] != mal:
        state["yummy"][str(yummy)] = {"mal_id": None, "status": "ambiguous", "fetched_at": now}
        for value in (mal, previous["mal_id"]):
            state["mal"][str(value)] = {"status": "ambiguous", "fetched_at": now}
        return
    state["yummy"][str(yummy)] = {"mal_id": mal, "status": "mapped" if mal else "missing", "fetched_at": now}
    if mal:
        old = state["mal"].get(str(mal))
        if old and old.get("yummy_id") not in (None, yummy):
            state["mal"][str(mal)] = {"status": "ambiguous", "fetched_at": now}
            for identity in (yummy, old["yummy_id"]):
                state["yummy"][str(identity)] = {"mal_id": None, "status": "ambiguous", "fetched_at": now}
        else:
            state["mal"][str(mal)] = {"yummy_id": yummy, "status": "mapped", "fetched_at": now}


def _validate_config(root, config):
    expected = {"vocabulary_path", "output_dir", "bootstrap_path", "supplement_path", "batch_size", "max_requests",
        "requests_per_second", "max_retries", "backoff_seconds", "max_sleep_seconds", "refresh_days", "negative_refresh_days"}
    if set(config) != expected:
        raise ValueError("Unexpected metadata configuration fields; tokens belong only in the environment")
    for key in ("max_requests", "batch_size", "max_retries"):
        if type(config[key]) is not int or config[key] < (0 if key == "max_retries" else 1):
            raise ValueError("Invalid integer request limit")
    if config["batch_size"] > 20:
        raise ValueError("Catalog batch size must be <=20")
    for key in ("requests_per_second", "backoff_seconds", "max_sleep_seconds", "refresh_days", "negative_refresh_days"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid rate/TTL value")
    for key, parent in (("vocabulary_path", "data"), ("output_dir", "data/processed/franchises")):
        if not (root / config[key]).resolve().is_relative_to((root / parent).resolve()):
            raise ValueError("Metadata paths outside designated workspace directory")
    for key in ("bootstrap_path", "supplement_path"):
        if config.get(key) and not (root / config[key]).resolve().is_relative_to(root / "configs"):
            raise ValueError("Bootstrap/supplement must be local configs")


def run_metadata_pipeline(root, config=None, execute=False, selected_mal_ids=None, client=None,
                          clock=time.time, sleep=time.sleep, allow_large_run=False):
    root = Path(root).resolve()
    config = dict(config or json.loads((root / "configs/viewing_order_metadata.json").read_text()))
    _validate_config(root, config)
    if execute and config["max_requests"] > 10 and not allow_large_run:
        raise ValueError("Large runs require explicit separate authorization and allow_large_run=True")
    vocabulary = pd.read_parquet(root / config["vocabulary_path"])
    all_mals = sorted(set(int(v) for v in vocabulary.myanimelist_id))
    selected = all_mals if selected_mal_ids is None else sorted(set(selected_mal_ids))
    if not selected or any(type(v) is not int or v <= 0 for v in selected):
        raise ValueError("Positive MAL IDs required")
    output = root / config["output_dir"]
    state_path = output / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"schema_version": 1,
        "mal": {}, "yummy": {}, "orders": {}, "errors": [], "total_requests": 0}
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    _bootstrap(state, root, config.get("bootstrap_path"))
    now = clock()
    missing = [m for m in selected if not _fresh(state["mal"].get(str(m)), config, now)]
    plan = {"dry_run": not execute, "train_items": len(all_mals), "selected_mal_ids": len(selected),
        "pending_mal_lookups": len(missing), "lookup_batches": math.ceil(len(missing) / config["batch_size"]),
        "max_requests": config["max_requests"], "requests": 0,
        "estimate_full_catalog": {"mal_lookup_batches_upper": math.ceil(len(all_mals) / config["batch_size"]),
            "detail_requests_upper_for_seeds": len(all_mals),
            "note": "Conservative seeds-only estimate; related external IDs/refresh/retries add requests; dedup reduces detail count"}}
    if not execute:
        return plan  # No client/token needed, no writes and no network.
    output.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        atomic_json(state_path, state)
    checkpoint()
    if client is None:
        import os
        from dotenv import dotenv_values
        token = os.environ.get("YUMMYANIME_APPLICATION_TOKEN") or dotenv_values(root / ".env").get("YUMMYANIME_APPLICATION_TOKEN")
        client = YummyAnimeClient(token)
    transport = MetadataRequests(config, state, checkpoint, clock, sleep)
    errors_at_start = len(state["errors"])
    def request(label, operation):
        try:
            return transport.call(label, operation)
        except YummyAnimeError as error:
            state["errors"].append({"operation": label, "http": error.status, "at": clock(),
                "temporary": isinstance(error, YummyAnimeNetworkError) or error.status in {408, 425, 429, 500, 502, 503, 504}})
            checkpoint()
            if error.status in (401, 403):
                raise RunStopped("authorization_error") from None
            raise
    stopped = None
    try:
        for start in range(0, len(missing), config["batch_size"]):
            batch = missing[start:start + config["batch_size"]]
            try:
                rows = request("catalog_by_mal", lambda: client.get_anime_by_mal_ids(batch))
            except YummyAnimeError as error:
                for mal in batch:
                    state["mal"][str(mal)] = {"status": "error", "fetched_at": clock(),
                        "retryable": isinstance(error, YummyAnimeNetworkError) or error.status in {408, 425, 429, 500, 502, 503, 504}}
                checkpoint()
                continue
            matches = {mal: set() for mal in batch}
            for row in rows:
                mal = (row.get("remote_ids") or {}).get("myanimelist_id")
                yummy = row.get("anime_id")
                if type(mal) is int and mal in matches and type(yummy) is int and yummy > 0:
                    matches[mal].add(yummy)
            for mal, found in matches.items():
                if len(found) == 1 and len(rows) < 20:
                    _record_resolution(state, next(iter(found)), mal, clock())
                else:
                    previous = state["mal"].get(str(mal), {})
                    for identity in found | ({previous["yummy_id"]} if previous.get("yummy_id") else set()):
                        state["yummy"][str(identity)] = {"mal_id": None, "status": "ambiguous" if found else "missing", "fetched_at": clock()}
                    state["mal"][str(mal)] = {"status": "ambiguous" if found or len(rows) >= 20 else "missing", "fetched_at": clock(), "candidate_yummy_ids": sorted(found)}
            checkpoint()
        attempted_members = set()
        for mal in selected:
            record = state["mal"].get(str(mal), {})
            if record.get("status") != "mapped":
                continue
            yummy = record["yummy_id"]
            known = {row["anime_id"] for order in state["orders"].values()
                     if _fresh(order, config, clock()) for row in order["viewing_order"]}
            if yummy in known or yummy in attempted_members or _fresh(state["orders"].get(str(yummy)), config, clock()):
                continue
            try:
                detail = request("anime_detail", lambda: client.get_anime(yummy))
            except YummyAnimeError as error:
                previous = state["orders"].get(str(yummy))
                state["orders"][str(yummy)] = {"seed_yummy_id": yummy, "viewing_order": [], "status": "error", "fetched_at": clock(),
                    "retryable": isinstance(error, YummyAnimeNetworkError) or error.status in {408, 425, 429, 500, 502, 503, 504}}
                if previous and previous["viewing_order"]:
                    state["orders"][str(yummy)].update({"viewing_order": previous["viewing_order"], "fetched_at": previous["fetched_at"]})
                    attempted_members.update(r["anime_id"] for r in previous["viewing_order"])
                checkpoint(); continue
            rows = detail.get("viewing_order")
            if detail.get("anime_id") != yummy:
                rows = None
                state["errors"].append({"operation": "detail_identity_mismatch", "http": 200, "at": clock()})
            if not isinstance(rows, list):
                rows = []
                state["errors"].append({"operation": "invalid_viewing_order", "http": 200, "at": clock()})
            clean = []
            valid = True
            for row in rows:
                data = row.get("data") or {}
                if type(row.get("anime_id")) is not int or row["anime_id"] <= 0 or type(data.get("id")) is not int or data["id"] <= 0 or type(data.get("index")) is not int or data["index"] < 0:
                    valid = False; break
                clean.append({"anime_id": row["anime_id"], "type": {"alias": (row.get("type") or {}).get("alias")},
                    "data": {"id": data["id"], "index": data["index"], "text": data.get("text") if isinstance(data.get("text"), str) else None}})
            if not valid:
                clean = []
                state["errors"].append({"operation": "invalid_order_item", "http": 200, "at": clock()})
            state["orders"][str(yummy)] = {"seed_yummy_id": yummy, "viewing_order": clean,
                "status": "ordered" if clean else "empty", "fetched_at": clock()}
            new_members = {row["anime_id"] for row in clean}
            for key, old in state["orders"].items():
                old_members = {row["anime_id"] for row in old["viewing_order"]}
                if key != str(yummy) and old_members and old_members <= new_members and not _fresh(old, config, clock()):
                    old["superseded_by"] = yummy
            attempted_members.update(r["anime_id"] for r in clean)
            checkpoint()
        related_ids = sorted({row["anime_id"] for order in state["orders"].values() if not order.get("superseded_by") for row in order["viewing_order"]
                              if not _fresh(state["yummy"].get(str(row["anime_id"])), config, clock())})
        for start in range(0, len(related_ids), config["batch_size"]):
            batch = related_ids[start:start + config["batch_size"]]
            try:
                rows = request("catalog_by_anime_id", lambda: client.get_anime_by_ids(batch))
            except YummyAnimeError:
                continue
            found = {}
            for row in rows:
                if row.get("anime_id") in batch:
                    found.setdefault(row["anime_id"], set()).add((row.get("remote_ids") or {}).get("myanimelist_id"))
            for yummy in batch:
                values = found.get(yummy, set())
                if len(values) <= 1 and len(rows) < 20:
                    _record_resolution(state, yummy, next(iter(values), None), clock())
                else:
                    state["yummy"][str(yummy)] = {"status": "ambiguous", "mal_id": None, "fetched_at": clock()}
            checkpoint()
    except RunStopped as error:
        stopped = str(error)
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        checkpoint()
    examples = [{"seed_yummy_id": order["seed_yummy_id"], "viewing_order": order["viewing_order"]}
                for order in state["orders"].values() if order["viewing_order"] and not order.get("superseded_by")]
    resolved = [{"anime_id": int(yummy), "myanimelist_id": row.get("mal_id")} for yummy, row in state["yummy"].items()]
    supplement = json.loads((root / config["supplement_path"]).read_text()) if config.get("supplement_path") and (root / config["supplement_path"]).exists() else None
    snapshot = merge_metadata_orders(examples, resolved, supplement)
    snapshot["source_fetch_times"] = sorted({datetime.fromtimestamp(order["fetched_at"], timezone.utc).isoformat() for order in state["orders"].values() if order["viewing_order"] and not order.get("superseded_by")})
    snapshot["retrieved_utc"] = max(snapshot["source_fetch_times"], default="1970-01-01T00:00:00+00:00")
    mapping = build_franchise_mapping(snapshot)
    covered = set(all_mals) & set(mapping.franchise)
    report = {**plan, "dry_run": False, "requests": transport.used, "total_requests": state["total_requests"],
        "stop_reason": stopped, "processed_selected_mal_ids": sum(str(m) in state["mal"] for m in selected),
        "known_franchise_groups": len(set(mapping.franchise.values())), "mapped_parts": len(snapshot["nodes"]),
        "covered_train_items": len(covered), "train_coverage": len(covered) / len(all_mals),
        "order_known_train_items": sum(mapping.ancestors(m) is not None for m in all_mals),
        "errors_this_run": sum(event["http"] != 200 for event in transport.events), "errors_total": len(state["errors"]),
        "failed_operations_this_run": len(state["errors"]) - errors_at_start,
        "ambiguous_identifiers": sum(r.get("status") == "ambiguous" for r in state["mal"].values()) + sum(r.get("status") == "ambiguous" for r in state["yummy"].values()),
        "conflicting_groups": sum(r["conflicting"] for r in snapshot["merge_quality"]),
        "unmapped_parts": sum(row["mal_id"] is None for order in snapshot["source_orders"] for row in order["items"]),
        "selected_status_counts": {status: sum(state["mal"].get(str(m), {}).get("status", "pending") == status for m in selected)
            for status in ("mapped", "missing", "ambiguous", "error", "pending")},
        "unresolved_seed_errors": sum(record.get("status") == "error" for record in state["mal"].values()),
        "unknown_order_groups": sum(not s["order_known"] for s in snapshot["ordered_sequences"]),
        "stale_orders": sum(not _fresh(r, config, clock()) for r in state["orders"].values() if not r.get("superseded_by")),
        "vocabulary_sha256": file_sha256(root / config["vocabulary_path"]), "mapping_fingerprint": mapping.fingerprint,
        "events": transport.events, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1, "completeness": "partial"}
    report["run_status"] = "stopped" if stopped else ("completed_with_errors" if report["errors_this_run"] or report["failed_operations_this_run"] or report["selected_status_counts"]["error"] else "completed")
    report["configuration"] = config
    # Publish a new immutable version, then atomically move a small local pointer.
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output / "versions" / version
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "mapping.json", snapshot)
    part_rows = []
    for order in snapshot["source_orders"]:
        quality = next(q for q in snapshot["merge_quality"] if q["franchise_id"] == order["franchise_id"])
        sequence = next(s for s in snapshot["ordered_sequences"] if s["franchise_id"] == order["franchise_id"])
        for row in order["items"]:
            fetched = max((source["fetched_at"] for source in state["orders"].values()
                if any(r["anime_id"] == row["yummy_id"] for r in source["viewing_order"])), default=clock())
            stage = next((i for i, step in enumerate(sequence["steps"]) if row["mal_id"] is not None and row["mal_id"] in step["mal_ids"]), None)
            alias = stage is not None and row["yummy_id"] not in sequence["steps"][stage]["yummy_ids"]
            role = "alternative" if alias or str(row["label"]).strip().casefold() == "альтернативная версия" else ("mainline" if stage is not None and sequence["order_known"] else row["role"])
            part_rows.append({"yummy_id": row["yummy_id"], "myanimelist_id": row["mal_id"],
                "franchise_id": order["franchise_id"], "viewing_order_index": row["index"],
                "part_role": role, "source_role": row["role"], "source_label": row["label"], "stage_group": stage,
                "part_type": next((n["format"] for n in snapshot["nodes"] if n["mal_id"] == row["mal_id"]), None),
                "source": "YummyAnime viewing_order", "retrieved_at": datetime.fromtimestamp(fetched, timezone.utc).isoformat(),
                "built_at": report["timestamp_utc"], "source_group_ids": json.dumps(quality["source_group_ids"]),
                "schema_version": 1, "order_known": sequence["order_known"], "quality": quality["status"],
                "completeness": "mal_resolved" if row["mal_id"] else "mal_unresolved"})
    parts = pd.DataFrame(part_rows, columns=["yummy_id", "myanimelist_id", "franchise_id", "viewing_order_index", "part_role", "source_role", "source_label", "stage_group", "part_type", "source", "retrieved_at", "built_at", "source_group_ids", "schema_version", "order_known", "quality", "completeness"])
    atomic_parquet(parts, directory / "parts.parquet")
    atomic_json(directory / "coverage.json", report)
    atomic_json(directory / "source_records.json", {"orders": state["orders"], "yummy_resolution": state["yummy"]})
    manifest = {"schema_version": 1, "files_sha256": {name: file_sha256(directory / name) for name in ("mapping.json", "parts.parquet", "coverage.json", "source_records.json")}}
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(output / "latest.json", {"schema_version": 1, "version": "versions/" + version,
        "manifest_sha256": file_sha256(directory / "manifest.json")})
    return report




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Default is a no-network, no-write dry-run")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--mal-ids", help="Optional diagnostic subset, comma-separated IDs; default: all train items")
    parser.add_argument("--allow-large-run", action="store_true", help="Only after separate user/provider approval")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/viewing_order_metadata.json").read_text())
    if args.max_requests is not None:
        config["max_requests"] = args.max_requests
    selected = [int(v) for v in args.mal_ids.split(",")] if args.mal_ids else None
    print(json.dumps(run_metadata_pipeline(root, config, args.execute, selected,
        allow_large_run=args.allow_large_run), indent=2))


if __name__ == "__main__":
    main()
