"""Step 11: assemble the golden record from the stage rows and run the nine validation checks (ADR section 6).

Flags are reported, never auto-fixed. Each flag: {code, severity (Error|Warning|Info), field, message}.
"""
import re
from datetime import date
from pipeline.step00_schema import VERSION, VOCAB
from pipeline.step09_damage import _numbers

_ID_PATTERNS = {  # check 1
    "header.claim_id": r"CLM-\d{6}", "policy_details.policy_number": r"POL-\d{9}",
    "incident_description.police_report_number": r"PR-\d{8}-\d{4}"}
_OBJECT_ID = r"PROP-\d+|LIAB-\d+|VIN\d{17}"
_ENUMS = (("header", "channel"), ("policy_details", "coverage_type"), ("incident_description", "incident_type"),
          ("incident_description", "location_type"))  # check 2 (object_type is checked per object)
_REQUIRED_ENUMS = {"channel", "incident_type", "location_type"}
_STAGES = ("01_ingest", "02_header", "03_claim_id", "04_policy", "05_policy_dates", "06_objects",
           "07_incident_date", "08_types", "09_damage", "10_police")


def flag(code, severity, field, message):
    return {"code": code, "severity": severity, "field": field, "message": message}


def _date(s):
    try:
        return date.fromisoformat(s) if s else None
    except (TypeError, ValueError):
        return None


def _pos_int(v):
    return type(v) is int and v > 0


def _trace(value, texts):
    return any(value in t for t in texts)


def validate(golden, texts=()):
    """Run checks 1-9 on a golden dict. `texts` = (raw_text, working text) for the traceability check."""
    out = []
    h, pol, objs, inc = (golden["header"], golden["policy_details"], golden["insured_objects"],
                         golden["incident_description"])
    sections = {"header": h, "policy_details": pol or {}, "incident_description": inc}

    for path, rx in _ID_PATTERNS.items():  # 1: IDs match their regex
        sec, key = path.split(".")
        v = sections[sec].get(key)
        if v is not None and not re.fullmatch(rx, v):
            out.append(flag("bad_id", "Error", path, f"{v!r} does not match {rx}"))
    for i, o in enumerate(objs or []):
        if o["object_id"] is not None and not re.fullmatch(_OBJECT_ID, o["object_id"]):
            out.append(flag("bad_id", "Error", f"insured_objects[{i}].object_id", f"{o['object_id']!r} is not PROP-/LIAB-/VIN"))

    for sec, key in _ENUMS:  # 2: enums in vocabulary (null is only allowed where the schema allows it)
        v = sections[sec].get(key)
        path = f"{sec}.{key}"
        if v is None:
            if key in _REQUIRED_ENUMS or (sec == "policy_details" and pol):
                out.append(flag("missing_required", "Error", path, "required enum is null"))
        elif v not in VOCAB[key]:
            out.append(flag("bad_enum", "Error", path, f"{v!r} is not in the vocabulary"))
    for i, o in enumerate(objs or []):
        if o["object_type"] not in VOCAB["object_type"]:
            out.append(flag("bad_enum", "Error", f"insured_objects[{i}].object_type", f"{o['object_type']!r}"))

    inc_d, rep_d = _date(h["incident_date"]), _date(h["report_date"])  # 3: incident <= report
    if inc_d and rep_d and inc_d > rep_d:
        out.append(flag("incident_after_report", "Error", "header.incident_date", f"{inc_d} is after {rep_d}"))

    eff, exp = (_date(pol.get("effective_date")), _date(pol.get("expiration_date"))) if pol else (None, None)
    if eff and exp and not eff < exp:  # 4: effective < expiration
        out.append(flag("bad_policy_window", "Error", "policy_details.effective_date", f"{eff} is not before {exp}"))

    cov = pol.get("coverage_type") if pol else None  # 5: coverage <-> object type
    for i, o in enumerate(objs or []):
        want = "Vehicle" if cov == "Auto" else "Building" if cov in ("Property", "Liability") else None
        if want and o["object_type"] != want:
            out.append(flag("coverage_object_mismatch", "Error", f"insured_objects[{i}].object_type",
                            f"{cov} coverage needs a {want}, got {o['object_type']}"))

    amounts = [("incident_description.estimated_damage_amount", inc["estimated_damage_amount"])] + \
              [(f"insured_objects[{i}].estimated_value", o["estimated_value"]) for i, o in enumerate(objs or [])]
    for path, v in amounts:  # 6: amounts are positive integers
        if v is not None and not _pos_int(v):
            out.append(flag("bad_amount", "Error", path, f"{v!r} is not a positive integer"))

    pr = inc["police_report_number"]  # 7: police report date part ~ incident date (+-3 days)
    if pr and inc_d and re.fullmatch(r"PR-\d{8}-\d{4}", pr):
        pr_d = _date(f"{pr[3:7]}-{pr[7:9]}-{pr[9:11]}")
        if pr_d is None or abs((pr_d - inc_d).days) > 3:
            out.append(flag("police_date_mismatch", "Warning", "incident_description.police_report_number",
                            f"{pr} vs incident {inc_d}"))

    if eff and exp and inc_d and not eff <= inc_d <= exp:  # 8: incident within the policy window
        out.append(flag("incident_outside_policy_window", "Info", "header.incident_date",
                        f"{inc_d} is outside {eff}..{exp}"))

    if texts:  # 9: every non-null value traceable to the text (labels and derived dates are exempt)
        digits = lambda s: re.sub(r"\D", "", s)
        for path, v in (("header.claim_id", h["claim_id"]), ("header.reported_by", h["reported_by"]),
                        ("policy_details.policy_number", pol and pol["policy_number"]),
                        ("policy_details.policyholder_name", pol and pol["policyholder_name"]),
                        ("incident_description.police_report_number", pr)):
            if v and not _trace(v, texts):
                out.append(flag("not_traceable", "Warning", path, f"{v!r} not found in the claim text"))
        for i, o in enumerate(objs or []):
            oid = o["object_id"]
            if oid and not _trace(digits(oid), [digits(t) for t in texts]):
                out.append(flag("not_traceable", "Warning", f"insured_objects[{i}].object_id", oid))
            if o["year"] and not any(re.search(rf"{o['year']}|[’‘']\s?{str(o['year'])[2:]}\b", t) for t in texts):
                out.append(flag("not_traceable", "Warning", f"insured_objects[{i}].year", str(o["year"])))
            if o["estimated_value"] and not any(o["estimated_value"] in _numbers(t) for t in texts):
                out.append(flag("not_traceable", "Warning", f"insured_objects[{i}].estimated_value", str(o["estimated_value"])))
            if o["location_address"] and not _trace(o["location_address"].split(",")[0], texts):
                out.append(flag("not_traceable", "Warning", f"insured_objects[{i}].location_address", o["location_address"]))
        d = inc["estimated_damage_amount"]
        if d and not any(d in _numbers(t) for t in texts):
            out.append(flag("not_traceable", "Warning", "incident_description.estimated_damage_amount", str(d)))
    return out


def assemble(rec_id, raw_text, stages):
    """Build the ADR section 7 record from {stage name: row}. Missing stages are Error flags, not silent nulls."""
    flags = [flag("stage_missing", "Error", s, f"stage {s} has no row") for s in _STAGES if s not in stages]
    g = lambda s: stages.get(s, {})
    ingest, hdr = g("01_ingest"), g("02_header")
    pol = g("04_policy").get("policy")
    dates = g("05_policy_dates").get("policy_dates") or {}
    types, damage = g("08_types"), g("09_damage")

    if ingest.get("truncated"):
        flags.append(flag("truncated", "Warning", "raw_text", "the claim text ends mid-sentence"))
    if hdr.get("channel_inferred"):
        flags.append(flag("channel_inferred", "Info", "header.channel", "no channel tag; inferred from wording"))
    if pol and pol.get("holder_inferred"):
        flags.append(flag("holder_inferred", "Info", "policy_details.policyholder_name", "holder differs from the caller"))
    for stage, row in (("07_incident_date", g("07_incident_date")), ("08_types", types), ("09_damage", damage)):
        for code in row.get("flags", []):
            sev = "Error" if code == "llm_unavailable" else "Warning"
            flags.append(flag(code, sev, stage, f"raised by {stage}"))

    golden = {
        "header": {"claim_id": g("03_claim_id").get("claim_id"), "report_date": hdr.get("report_date"),
                   "incident_date": g("07_incident_date").get("incident_date"),
                   "reported_by": hdr.get("reported_by"), "channel": hdr.get("channel")},
        "policy_details": None if pol is None else {
            "policy_number": pol["policy_number"], "policyholder_name": pol["policyholder_name"],
            "coverage_type": pol["coverage_type"], "effective_date": dates.get("effective_date"),
            "expiration_date": dates.get("expiration_date")},
        "insured_objects": None if pol is None else g("06_objects").get("insured_objects"),
        "incident_description": {
            "incident_type": types.get("incident_type"), "location_type": types.get("location_type"),
            "estimated_damage_amount": damage.get("estimated_damage_amount"),
            "police_report_number": g("10_police").get("police_report_number")}}
    flags += validate(golden, (raw_text, ingest.get("text", raw_text)))
    return {"id": rec_id, "version": VERSION, "raw_text": raw_text, "golden": golden, "flags": flags,
            "review_notes": ""}
