"""Fine-tuning prep, step 1 (FR-15): audit the pipeline golden records against the dataset ground_truth.

The dataset labels are the training targets; the pipeline output is an independent second opinion. This module
only reads both and writes, under data/audit/:
    field_agreement.json   per-field agreement counts
    conflicts.jsonl        one row per disagreement: kind = conflict | pipeline_null | dataset_null
    profile.json           class balance, personal-data inventory, text lengths
"""
import json
import re
from collections import Counter
from pathlib import Path
from pipeline.store import write_stage

ROOT = Path(__file__).resolve().parent.parent
_SCALARS = {
    "header": ("claim_id", "report_date", "incident_date", "reported_by", "channel"),
    "policy_details": ("policy_number", "policyholder_name", "coverage_type", "effective_date", "expiration_date"),
    "incident_description": ("incident_type", "location_type", "estimated_damage_amount", "police_report_number")}
_OBJECT_FIELDS = ("object_id", "object_type", "make_model", "year", "location_address", "estimated_value")


def _kind(pipeline, dataset):
    if pipeline == dataset:
        return None
    if pipeline is None:
        return "pipeline_null"
    return "dataset_null" if dataset is None else "conflict"


def _objects(golden):
    objs = golden.get("insured_objects")
    return None if objs is None else sorted(objs, key=lambda o: (o.get("object_id") or "", str(o.get("year"))))


def compare(golden_rows, raw):
    """-> (agreement {field: {agree, total}}, conflicts [row]). Rows are aligned by position; ids come from golden."""
    if len(golden_rows) != len(raw):
        raise ValueError(f"{len(golden_rows)} golden records but {len(raw)} raw records")
    agree, conflicts = {}, []

    def note(rec, field, pipeline, dataset, count_as=None):
        a = agree.setdefault(count_as or field, {"agree": 0, "total": 0})
        a["total"] += 1
        kind = _kind(pipeline, dataset)
        if kind is None:
            a["agree"] += 1
        else:
            conflicts.append({"id": f"{rec}:{field}", "record": rec, "field": field, "kind": kind,
                              "pipeline": pipeline, "dataset": dataset})

    for row, item in zip(golden_rows, raw):
        rec, g, t = row["id"], row["golden"], item["ground_truth"]
        for sec, keys in _SCALARS.items():
            for k in keys:
                note(rec, f"{sec}.{k}", (g.get(sec) or {}).get(k), (t.get(sec) or {}).get(k))
        go, to = _objects(g), _objects(t)
        if go is None and to is None:
            continue
        if go is None or to is None or len(go) != len(to):
            kind = _kind(go, to) if (go is None or to is None) else "conflict"
            conflicts.append({"id": f"{rec}:insured_objects", "record": rec, "field": "insured_objects",
                              "kind": kind, "pipeline": go, "dataset": to})
            continue
        for i, (a, b) in enumerate(zip(go, to)):
            for k in _OBJECT_FIELDS:
                note(rec, f"insured_objects[{i}].{k}", a.get(k), b.get(k), count_as=f"insured_objects.{k}")
    return agree, conflicts


def profile(raw):
    """Class balance, personal-data inventory and text statistics of the dataset labels and texts."""
    labels = [item["ground_truth"] for item in raw]
    texts = [item["claim_text"] for item in raw]
    count = lambda get: dict(sorted(Counter(get(t) for t in labels).items(), key=lambda kv: str(kv[0])))
    incident = count(lambda t: t["incident_description"]["incident_type"])
    names = set()
    objs = []
    for t in labels:
        names.add(t["header"]["reported_by"])
        if t.get("policy_details"):
            names.add(t["policy_details"]["policyholder_name"])
        objs += t.get("insured_objects") or []
    first = Counter(n.split()[0] for n in names if n)
    ids = lambda key, sec: sum((t.get(sec) or {}).get(key) is not None for t in labels)
    joined = "\n".join(texts)
    lens = [len(x) for x in texts]
    return {
        "records": len(raw),
        "label_balance": {
            "incident_type": incident,
            "location_type": count(lambda t: t["incident_description"]["location_type"]),
            "channel": count(lambda t: t["header"]["channel"]),
            "coverage_type": count(lambda t: (t.get("policy_details") or {}).get("coverage_type") or "none")},
        "rare_incident_types": sorted(k for k, v in incident.items() if v <= 2),
        "text_chars": {"min": min(lens), "avg": sum(lens) // len(lens), "max": max(lens)},
        "duplicate_texts": len(texts) - len(set(texts)),
        "personal_data": {
            "labels": {
                "distinct_person_names": len(names),
                "repeated_first_names": {k: v for k, v in sorted(first.items()) if v > 1},
                "street_addresses": sum(o.get("location_address") is not None for o in objs),
                "vins": sum((o.get("object_id") or "").startswith("VIN") for o in objs),
                "policy_numbers": ids("policy_number", "policy_details"),
                "claim_ids": ids("claim_id", "header"),
                "police_report_numbers": ids("police_report_number", "incident_description")},
            "text_scan": {
                "email": len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", joined)),
                "phone": len(re.findall(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", joined)),
                "ssn_like": len(re.findall(r"\b\d{3}-\d{2}-\d{4}\b", joined))}}}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def run(golden_path=ROOT / "data" / "final" / "golden.jsonl", raw_path=ROOT / "insurance_claims.json",
        out_dir=ROOT / "data" / "audit"):
    """Write the three audit files; returns {'field_agreement', 'conflicts', 'profile'} paths."""
    golden_rows = [json.loads(x) for x in Path(golden_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    agree, conflicts = compare(golden_rows, raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"records": len(raw), "conflicts": sum(c["kind"] == "conflict" for c in conflicts),
              "pipeline_null": sum(c["kind"] == "pipeline_null" for c in conflicts),
              "dataset_null": sum(c["kind"] == "dataset_null" for c in conflicts), "fields": agree}
    return {"field_agreement": _write_json(out / "field_agreement.json", report),
            "conflicts": write_stage(out, "conflicts", conflicts),
            "profile": _write_json(out / "profile.json", profile(raw))}
