"""Fine-tuning prep, step 4 (FR-18): chat-style training examples.

One example per de-identified claim:
    system     the task, the schema, every allowed value and the rules (built from the frozen step-0 constants)
    user       the claim text
    assistant  the JSON record, compact, in schema key order
Every target is validated against the schema first; a bad one is rejected with its reasons and never exported.
examples.jsonl holds only {"messages": [...]} - no flags, notes, provenance or internal ids. The record ids live in
examples_index.json (same order) so the split step can keep a manifest.
"""
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from pipeline.step00_schema import SCHEMA, VOCAB

_ID = {"claim_id": r"CLM-\d{6}", "policy_number": r"POL-\d{9}", "police_report_number": r"PR-\d{8}-\d{4}",
       "object_id": r"(?:PROP|LIAB)-\d+|VIN\d{17}"}
_ENUM_REQUIRED = {"channel", "incident_type", "location_type", "object_type"}


def build_system_prompt():
    schema = {sec: ([dict(f)] if sec == "insured_objects" else dict(f)) for sec, f in SCHEMA.items()}
    allowed = "\n".join(f"- {field}: {' | '.join(values)}" for field, values in VOCAB.items())
    return (
        "You extract structured data from one insurance claim report. Reply with ONE JSON object and nothing else.\n\n"
        "Schema (use null for anything that is not stated or derivable - never guess):\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Allowed values:\n{allowed}\n\n"
        "Rules:\n"
        "- Dates are ISO YYYY-MM-DD; resolve a relative date (\"yesterday\", \"last year\") against the report date.\n"
        "- Amounts are whole numbers with no $ or commas.\n"
        "- Ids: claim_id CLM-nnnnnn, policy_number POL-nnnnnnnnn, police_report_number PR-nnnnnnnn-nnnn, "
        "object_id PROP-n, LIAB-n or VIN followed by 17 digits.\n"
        "- If the claim states no valid policy number, policy_details and insured_objects are null.\n"
        "- List every insured object on the policy, even one that was not involved in the incident.\n"
        "- estimated_damage_amount is a repair or replacement estimate (prefer a written exact figure); "
        "estimated_value is the value of an insured object.\n"
        "- Use only the allowed values above; never invent a label.")


SYSTEM_PROMPT = build_system_prompt()


def _iso(v):
    try:
        return isinstance(v, str) and date.fromisoformat(v) is not None
    except ValueError:
        return False


def _text(v):
    return isinstance(v, str) and bool(v.strip())


def validate_target(t):
    """Problems (strings) found in one target; an empty list means it is a valid record."""
    out = []
    if not isinstance(t, dict) or set(t) != set(SCHEMA):
        return [f"sections must be exactly {sorted(SCHEMA)}"]

    def keys(path, d, expected):
        if not isinstance(d, dict) or set(d) != set(expected):
            out.append(f"{path}: keys must be exactly {sorted(expected)}")
            return False
        return True

    def check(path, v, kind, nullable=True):
        k = path.split(".")[-1]
        if v is None:
            if not nullable:
                out.append(f"{path}: required, got null")
        elif kind == "text" and not _text(v):
            out.append(f"{path}: must be a non-empty string")
        elif kind == "date" and not _iso(v):
            out.append(f"{path}: {v!r} is not an ISO date")
        elif kind == "int" and not (type(v) is int and v > 0):
            out.append(f"{path}: {v!r} is not a positive integer")
        elif kind == "enum" and v not in VOCAB[k]:
            out.append(f"{path}: {v!r} is not an allowed value")
        elif kind == "id" and not re.fullmatch(_ID[k], str(v)):
            out.append(f"{path}: {v!r} is not a valid {k}")

    if keys("header", t["header"], SCHEMA["header"]):
        h = t["header"]
        check("header.claim_id", h["claim_id"], "id")
        check("header.report_date", h["report_date"], "date", nullable=False)
        check("header.incident_date", h["incident_date"], "date")
        check("header.reported_by", h["reported_by"], "text", nullable=False)
        check("header.channel", h["channel"], "enum", nullable=False)
    p, objs = t["policy_details"], t["insured_objects"]
    if (p is None) != (objs is None):
        out.append("policy_details and insured_objects must both be null or both be present")
    if p is not None and keys("policy_details", p, SCHEMA["policy_details"]):
        check("policy_details.policy_number", p["policy_number"], "id", nullable=False)
        check("policy_details.policyholder_name", p["policyholder_name"], "text", nullable=False)
        check("policy_details.coverage_type", p["coverage_type"], "enum", nullable=False)
        check("policy_details.effective_date", p["effective_date"], "date")
        check("policy_details.expiration_date", p["expiration_date"], "date")
    if objs is not None:
        if not isinstance(objs, list):
            out.append("insured_objects: must be a list")
        for i, o in enumerate(objs if isinstance(objs, list) else []):
            if keys(f"insured_objects[{i}]", o, SCHEMA["insured_objects"]):
                pre = f"insured_objects[{i}]."
                check(pre + "object_id", o["object_id"], "id")
                check(pre + "object_type", o["object_type"], "enum", nullable=False)
                check(pre + "make_model", o["make_model"], "text")
                check(pre + "year", o["year"], "int")
                check(pre + "location_address", o["location_address"], "text")
                check(pre + "estimated_value", o["estimated_value"], "int")
    if keys("incident_description", t["incident_description"], SCHEMA["incident_description"]):
        d = t["incident_description"]
        check("incident_description.incident_type", d["incident_type"], "enum", nullable=False)
        check("incident_description.location_type", d["location_type"], "enum", nullable=False)
        check("incident_description.estimated_damage_amount", d["estimated_damage_amount"], "int")
        check("incident_description.police_report_number", d["police_report_number"], "id")
    return out


def _canonical(t):
    out = {}
    for sec, fields in SCHEMA.items():
        v = t[sec]
        out[sec] = (None if v is None else [{k: o[k] for k in fields} for o in v] if sec == "insured_objects"
                    else {k: v[k] for k in fields})
    return out


def build_examples(rows):
    """-> (record ids, examples, rejected). Rows are de-identified {id, claim_text, target, ...}."""
    ids, examples, rejected = [], [], []
    for r in rows:
        problems = validate_target(r["target"])
        if problems:
            rejected.append({"id": r["id"], "problems": problems})
            continue
        answer = json.dumps(_canonical(r["target"]), ensure_ascii=False, separators=(",", ":"))
        ids.append(r["id"])
        examples.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": r["claim_text"]},
                                      {"role": "assistant", "content": answer}]})
    return ids, examples, rejected


def _span(values):
    return {"min": min(values), "avg": sum(values) // len(values), "max": max(values)} if values else {}


def write_examples(rows, out_dir):
    """Write examples.jsonl and examples_index.json into out_dir -> ({'examples', 'index'} paths, rejected)."""
    ids, examples, rejected = build_examples(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True, ensure_ascii=False) + "\n")
    user = [len(e["messages"][1]["content"]) for e in examples]
    asst = [len(e["messages"][2]["content"]) for e in examples]
    index = {"ids": ids, "rejected": rejected, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
             "stats": {"examples": len(examples), "user_chars": _span(user), "assistant_chars": _span(asst),
                       # rough: about 4 characters per token, system prompt included
                       "approx_tokens_per_example": _span([(len(SYSTEM_PROMPT) + u + a) // 4 for u, a in zip(user, asst)])}}
    (out / "examples_index.json").write_text(json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    return {"examples": out / "examples.jsonl", "index": out / "examples_index.json"}, rejected
