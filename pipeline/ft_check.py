"""Fine-tuning prep, step 6 (FR-20): independent readiness checks on the FINAL files.

The checker works from the files in data/finetune and from the original (unmasked) records, and re-derives each
verdict itself - it does not ask the de-identification code whether masking worked. Verdicts:
    pass | pass_with_warnings | fail            (a check is pass | warn | fail)
Every detail string is free of original personal values, so the report is safe to keep next to the data.
"""
import hashlib
import json
import re
import tempfile
from pathlib import Path
from pipeline import ft_deid, ft_export, ft_split
from pipeline.step11_validate import validate

ROOT = Path(__file__).resolve().parent.parent
FILES = ("train.jsonl", "validation.jsonl", "examples.jsonl", "examples_index.json", "split_manifest.json")
CHECKS = ["file_shape", "schema", "traceability", "personal_data", "duplicates", "split", "exclusions",
          "determinism", "size"]
_SUFFIX = "(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Blvd|Boulevard|Ct|Court|Way|Pl|Place)"
_GENERIC_PII = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "phone": r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b",
                "ssn_like": r"\b\d{3}-\d{2}-\d{4}\b"}


def _res(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def _jl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _key(example):
    return json.dumps(example, sort_keys=True, ensure_ascii=False)


def _answer(example):
    return json.loads(example["messages"][2]["content"])


def _originals(rows):
    """Personal values of the original records, as match-ready sets (never written anywhere)."""
    names, digits, houses, streets, police = set(), set(), set(), set(), set()
    for r in rows:
        t = r["target"]
        h, p, inc = t["header"], t.get("policy_details") or {}, t["incident_description"]
        for n in (h.get("reported_by"), p.get("policyholder_name")):
            names.update(x for x in (n or "").split() if len(x) >= 3)
        for v in (h.get("claim_id"), p.get("policy_number")):
            if v:
                digits.add(v.split("-")[1])
        if inc.get("police_report_number"):
            police.add(inc["police_report_number"])
        for o in t.get("insured_objects") or []:
            oid = o.get("object_id") or ""
            if oid.startswith("VIN"):
                digits.add(oid[3:])
            elif "-" in oid:
                digits.add(oid.split("-")[1])
            toks = (o.get("location_address") or "").split(",")[0].split()
            if toks and toks[0].isdigit():
                if len(toks[0]) >= 4:
                    houses.add(toks[0])
                streets.add(" ".join(toks[1:-1] if len(toks) > 2 and re.fullmatch(_SUFFIX, toks[-1]) else toks[1:]))
    return names, digits | houses, streets, police


def _leaks(text, originals):
    names, nums, streets, police = originals
    kinds = set()
    if any(re.search(rf"\b{re.escape(n)}\b", text) for n in names):
        kinds.add("name")
    if any(re.search(rf"(?<![\d$])(?<!\d,){d}(?!\d|,\d)", text) for d in nums):
        kinds.add("number")
    if any(re.search(rf"\b{re.escape(s)}\s+{_SUFFIX}\b|\b\d+\s+{re.escape(s)}\b", text) for s in streets if s):
        kinds.add("street")
    if any(p in text for p in police):
        kinds.add("police_report")
    kinds.update(k for k, rx in _GENERIC_PII.items() if re.search(rx, text))
    return kinds


def _untraceable(target, text):
    return {f["field"] for f in validate(target, (text,)) if f["code"] == "not_traceable"}


def _report(checks, dataset, warnings, deid_report):
    fails = any(c["status"] == "fail" for c in checks)
    return {"verdict": "fail" if fails else "pass_with_warnings" if warnings or any(c["status"] == "warn" for c in checks)
            else "pass", "checks": checks, "dataset": dataset, "warnings": warnings,
            "residual_candidates": (deid_report or {}).get("residual_candidates", {})}


def run_checks(finetune_dir, reconciled_rows, excluded_rows, deid_report=None):
    d = Path(finetune_dir)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        checks = [_res(n, "fail", "missing file(s): " + ", ".join(missing) if n == "file_shape" else "not run: input files missing")
                  for n in CHECKS]
        return _report(checks, {}, [], deid_report)
    train, val, examples = _jl(d / "train.jsonl"), _jl(d / "validation.jsonl"), _jl(d / "examples.jsonl")
    index = json.loads((d / "examples_index.json").read_text(encoding="utf-8"))
    manifest = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    final = train + val
    rows = sorted(reconciled_rows, key=lambda r: r["id"])
    by_id = {r["id"]: r for r in rows}
    id_of = {_key(e): i for i, e in zip(index["ids"], examples)}
    checks, warnings = [], []

    # 1 file shape: id-free, three messages, our system prompt, no test set
    bad = []
    if [p.name for p in d.glob("test*")]:
        bad.append("a test-set file exists (the split has no test set)")
    for e in final:
        roles = [m.get("role") for m in e.get("messages", [])]
        if set(e) != {"messages"}:
            bad.append("an example has keys besides 'messages'")
        elif roles != ["system", "user", "assistant"]:
            bad.append(f"unexpected message roles {roles}")
        elif e["messages"][0]["content"] != ft_export.SYSTEM_PROMPT:
            bad.append("an example's system prompt is not the exported one")
    text_all = (d / "train.jsonl").read_text(encoding="utf-8") + (d / "validation.jsonl").read_text(encoding="utf-8")
    if re.search(r"rec-\d{3}", text_all):
        bad.append("an internal record id ('rec-nnn') is in a training file")
    checks.append(_res("file_shape", "fail" if bad else "pass", "; ".join(sorted(set(bad))) or
                       f"{len(final)} examples, 3 messages each, only 'messages' keys, no ids, no test set"))

    # 2 schema + vocabulary
    problems = []
    for n, e in enumerate(final):
        try:
            problems += [f"example {n}: {p}" for p in ft_export.validate_target(_answer(e))]
        except (ValueError, KeyError, IndexError, TypeError):
            problems.append(f"example {n}: the answer is not valid JSON")
    checks.append(_res("schema", "fail" if problems else "pass", "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
                       if problems else "every answer is a valid record: schema, enums, id formats, dates, numbers"))

    # 3 traceability: a label found in the text before masking must still be found after
    induced, before_total, unknown = [], 0, 0
    for e in final:
        i = id_of.get(_key(e))
        if i is None or i not in by_id:
            unknown += 1
            continue
        try:
            before = _untraceable(by_id[i]["target"], by_id[i]["claim_text"])
            after = _untraceable(_answer(e), e["messages"][1]["content"])
        except (ValueError, KeyError, TypeError):
            unknown += 1
            continue
        before_total += len(before)
        induced += [f"{f}" for f in sorted(after - before)]
    bad = bool(induced or unknown)
    checks.append(_res("traceability", "fail" if bad else "pass",
                       (f"masking cut {len(induced)} label(s) loose from their text ({', '.join(sorted(set(induced))[:5])})" if induced else "")
                       + (f"{' ' if induced else ''}{unknown} final example(s) could not be matched to a source record" if unknown else "")
                       if bad else "every label that was found in its text before masking is still found after",
                       not_in_text_before_masking=before_total))
    if before_total:
        warnings.append(f"{before_total} label value(s) were never literally in the claim text even before masking "
                        "(dataset label issue, e.g. an inferred holder or an address assembled from fragments).")

    # 4 personal data: nothing original anywhere under data/finetune
    originals, found = _originals(rows), {}
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        kinds = _leaks(f.read_text(encoding="utf-8", errors="replace"), originals)
        if kinds:
            found[f.name] = sorted(kinds)
    checks.append(_res("personal_data", "fail" if found else "pass",
                       "; ".join(f"{k}: {', '.join(v)}" for k, v in found.items()) if found else
                       "no original name, id, VIN, house number, street or police number, and no email/phone/SSN, in any file"))
    if deid_report and deid_report.get("residual_candidates"):
        warnings.append(f"{len(deid_report['residual_candidates'])} capitalised words could not be classified by the "
                        "de-identification rules (residual candidates in the report): a person should glance at them.")

    # 5 duplicates
    users = [e["messages"][1]["content"] for e in final]
    dup = len(users) - len(set(users))
    checks.append(_res("duplicates", "fail" if dup else "pass", f"{dup} duplicate claim text(s) across train and validation"
                       if dup else "no claim text appears twice"))

    # 6 split: disjoint, complete, 80/20, stratification rules
    tk, vk = [_key(e) for e in train], [_key(e) for e in val]
    probs = []
    if set(tk) & set(vk):
        probs.append(f"{len(set(tk) & set(vk))} example(s) in both train and validation")
    if sorted(tk + vk) != sorted(_key(e) for e in examples) and not manifest.get("dropped_duplicates"):
        probs.append("train + validation is not exactly the exported examples")
    n = len(final)
    want = int(0.2 * n + 0.5)
    if abs(len(val) + manifest.get("shortfall", 0) - want) > 1:
        probs.append(f"validation has {len(val)} of {n} examples; 20% is {want}")
    types = lambda es: [_answer(e)["incident_description"]["incident_type"] for e in es]
    try:
        all_t, tr_t, va_t = types(final), types(train), types(val)
        for t in set(all_t):
            if all_t.count(t) == 1 and t in va_t:
                probs.append("a single-example type is in validation")
            if all_t.count(t) >= 2 and t not in tr_t:
                probs.append("a type has no training example")
    except (ValueError, KeyError, IndexError, TypeError):
        probs.append("could not read incident types")
    if (len(manifest.get("train_ids", [])), len(manifest.get("validation_ids", []))) != (len(train), len(val)):
        probs.append("the manifest counts differ from the files")
    checks.append(_res("split", "fail" if probs else "pass", "; ".join(sorted(set(probs))) if probs else
                       f"train {len(train)} / validation {len(val)} = {len(val) / n:.0%} validation, disjoint, complete, no test set"))

    # 7 exclusions: a record with an unresolved conflict is never exported
    exported = set(manifest.get("train_ids", [])) | set(manifest.get("validation_ids", []))
    leaked = sorted(x["id"] for x in excluded_rows if x["id"] in exported)
    checks.append(_res("exclusions", "fail" if leaked else "pass", f"excluded record(s) were exported: {', '.join(leaked)}"
                       if leaked else f"none of the {len(excluded_rows)} excluded record(s) is in the export"))

    # 8 determinism: rebuilding from the same inputs reproduces the files byte for byte
    with tempfile.TemporaryDirectory() as tmp:
        masked, _ = ft_deid.deidentify(rows)
        ft_export.write_examples(masked, tmp)
        ft_split.write_split(tmp)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        differ = [f for f in FILES if sha(Path(tmp) / f) != sha(d / f)]
    checks.append(_res("determinism", "fail" if differ else "pass", "differs from a fresh rebuild: " + ", ".join(differ)
                       if differ else "a fresh rebuild from the original records is byte-identical"))

    # 9 size: always stated, warns when too small to fine-tune well
    if len(train) < 100:
        msg = (f"{len(train)} training and {len(val)} validation examples is too small to fine-tune a model well "
               "(extraction usually needs hundreds to thousands). Use it as an evaluation set first, or add many more claims.")
    elif len(train) < 1000:
        msg = f"{len(train)} training examples is small; expect modest gains and watch the validation loss."
    else:
        msg = ""
    checks.append(_res("size", "warn" if msg else "pass", msg or f"{len(train)} training examples"))
    if msg:
        warnings.insert(0, msg)
    singles = manifest.get("single_example_types", [])
    if singles:
        warnings.append(f"{len(singles)} incident type(s) have a single example (kept in train, never validated): "
                        + ", ".join(singles))
    dataset = {"examples": n, "train": len(train), "validation": len(val), "excluded_conflicts": len(excluded_rows),
               **{k: v for k, v in index.get("stats", {}).items() if k == "approx_tokens_per_example"}}
    return _report(checks, dataset, warnings, deid_report)


def run(out_dir=ROOT / "data"):
    """Check <out_dir>/finetune against the originals in <out_dir>/review; write audit/readiness_report.json."""
    out = Path(out_dir)
    rows = _jl(out / "review" / "reconciled.jsonl")
    excluded = _jl(out / "review" / "excluded.jsonl") if (out / "review" / "excluded.jsonl").exists() else []
    deid_path = out / "audit" / "deid_report.json"
    deid = json.loads(deid_path.read_text(encoding="utf-8")) if deid_path.exists() else None
    report = run_checks(out / "finetune", rows, excluded, deid)
    path = out / "audit" / "readiness_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path, report
