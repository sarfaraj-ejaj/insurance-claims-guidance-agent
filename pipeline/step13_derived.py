"""Derived layer (FR-14): is_active_at_incident, kept in its own file so the golden schema stays frozen.

true  if effective_date <= incident_date <= expiration_date (boundaries inclusive)
false if the incident falls outside that window
null  if any of the three dates is missing or unusable
"""
from datetime import date
from pipeline.store import write_stage


def _date(s):
    try:
        return date.fromisoformat(s) if s else None
    except (TypeError, ValueError):
        return None


def is_active_at_incident(golden):
    pol = golden["policy_details"] or {}
    inc, eff, exp = (_date(golden["header"]["incident_date"]), _date(pol.get("effective_date")),
                     _date(pol.get("expiration_date")))
    if None in (inc, eff, exp):
        return None
    return eff <= inc <= exp


def write_derived(records, out_dir):
    """Write <out_dir>/derived.jsonl with one {id, claim_id, is_active_at_incident} row per record."""
    rows = [{"id": r["id"], "claim_id": r["golden"]["header"]["claim_id"],
             "is_active_at_incident": is_active_at_incident(r["golden"])} for r in records]
    return write_stage(out_dir, "derived", rows)
