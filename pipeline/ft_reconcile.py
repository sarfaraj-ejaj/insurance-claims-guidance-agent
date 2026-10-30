"""Fine-tuning prep, step 2 (FR-16): one training target per claim, with per-field provenance.

The target is the dataset label (ground_truth is only read). The pipeline output is the second opinion:
    agree        both sources give the same value
    dataset_only the pipeline has none because a Claude step has not run (type / location / damage)
    pipeline_only the dataset has none but the pipeline found a value (target stays null, warning)
    decision     a human settled a conflict in data/review/decisions.jsonl
    conflict     anything else that differs - the record is EXCLUDED until a decision exists

decisions.jsonl, one JSON object per line:
    {"record": "rec-008", "field": "policy_details.coverage_type", "choose": "dataset" | "pipeline" | "value",
     "value": ..., "by": "name", "reason": "why"}        ("value" is required only for choose = "value")
"""
import copy
import json
import re
from pathlib import Path
from pipeline import ft_audit
from pipeline.store import write_stage

ROOT = Path(__file__).resolve().parent.parent
LLM_FIELDS = {"incident_description.incident_type", "incident_description.location_type",
              "incident_description.estimated_damage_amount"}
_OBJECT_PATH = re.compile(rf"insured_objects\[(\d+)\]\.({'|'.join(ft_audit._OBJECT_FIELDS)})")
_SCALAR_PATHS = {f"{sec}.{k}" for sec, keys in ft_audit._SCALARS.items() for k in keys}
_objects = ft_audit._objects


def load_decisions(path):
    """Decisions from a JSONL file; a missing file means none. Bad JSON names the line."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path.name} line {n}: not valid JSON ({e})") from e
    return out


def _check_decision(d, known_records):
    for key in ("record", "field", "choose", "by", "reason"):
        if not (isinstance(d.get(key), str) and d[key].strip()):
            raise ValueError(f"decision needs a non-empty {key!r}: {d}")
    if d["choose"] not in ("pipeline", "dataset", "value"):
        raise ValueError(f"choose must be pipeline, dataset or value: {d}")
    if d["choose"] == "value" and "value" not in d:
        raise ValueError(f"choose = value needs a 'value': {d}")
    if d["record"] not in known_records:
        raise ValueError(f"unknown record {d['record']!r}")
    f = d["field"]
    if f not in _SCALAR_PATHS and f != "insured_objects" and not _OBJECT_PATH.fullmatch(f):
        raise ValueError(f"unknown field {f!r}")


def _set(target, field, value):
    if field == "insured_objects":
        target["insured_objects"] = copy.deepcopy(value)
    elif (m := _OBJECT_PATH.fullmatch(field)):
        target["insured_objects"][int(m.group(1))][m.group(2)] = value
    else:
        sec, key = field.split(".")
        target[sec][key] = value


def _llm_missing(row):
    return any(f.get("code") in ("llm_unavailable", "stage_missing") for f in row.get("flags", []))


def _paths(target):
    out = [f"{sec}.{k}" for sec, keys in ft_audit._SCALARS.items() for k in keys]
    for i, _ in enumerate(target.get("insured_objects") or []):
        out += [f"insured_objects[{i}].{k}" for k in ft_audit._OBJECT_FIELDS]
    return out


def reconcile(golden_rows, raw, decisions):
    """-> (ready rows, excluded rows, ignored decisions)."""
    _, conflicts = ft_audit.compare(golden_rows, raw)
    ids = {r["id"] for r in golden_rows}
    for d in decisions:
        _check_decision(d, ids)
    chosen = {(d["record"], d["field"]): d for d in decisions}
    by_rec = {}
    for c in conflicts:
        by_rec.setdefault(c["record"], []).append(c)
    used = set()
    ready, excluded = [], []

    for row, item in zip(golden_rows, raw):
        rid = row["id"]
        target = copy.deepcopy(item["ground_truth"])
        if target.get("insured_objects"):
            target["insured_objects"] = _objects(target)
        prov = {p: "agree" for p in _paths(target)}
        warnings, unresolved = [], []
        for c in by_rec.get(rid, []):
            field, kind = c["field"], c["kind"]
            if kind == "dataset_null":
                source = "pipeline_only"
                warnings.append(f"pipeline_only:{field}")
            elif kind == "pipeline_null" and field in LLM_FIELDS and _llm_missing(row):
                source = "dataset_only"
            else:
                d = chosen.get((rid, field))
                if d is None:
                    unresolved.append({k: c[k] for k in ("field", "kind", "pipeline", "dataset")})
                    source = "conflict"
                else:
                    used.add((rid, field))
                    value = {"pipeline": c["pipeline"], "dataset": c["dataset"], "value": d.get("value")}[d["choose"]]
                    _set(target, field, value)
                    source = "decision"
            prov[field] = source
        if unresolved:
            excluded.append({"id": rid, "reasons": unresolved})
        else:
            ready.append({"id": rid, "claim_text": item["claim_text"], "target": target,
                          "provenance": prov, "warnings": sorted(warnings)})
    ignored = [{"record": r, "field": f, "why": "no conflict on this field"}
               for (r, f) in sorted(chosen) if (r, f) not in used]
    return ready, excluded, ignored


def run(golden_path=ROOT / "data" / "final" / "golden.jsonl", raw_path=ROOT / "insurance_claims.json",
        decisions_path=ROOT / "data" / "review" / "decisions.jsonl", out_dir=ROOT / "data" / "finetune"):
    """Write reconciled.jsonl (ready records) and excluded.jsonl; returns their paths."""
    golden_rows = [json.loads(x) for x in Path(golden_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    ready, excluded, _ = reconcile(golden_rows, raw, load_decisions(decisions_path))
    return {"reconciled": write_stage(out_dir, "reconciled", ready), "excluded": write_stage(out_dir, "excluded", excluded)}
