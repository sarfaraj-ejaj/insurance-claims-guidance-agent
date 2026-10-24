"""Step 12: write the final golden records and the reviewer's flags report (ADR section 7, AC-8).

    golden.jsonl       one ADR-shaped record per line: id, version, raw_text, golden, flags, review_notes
    flags_report.json  every record with at least one flag, worst severity first, plus summary counts

Re-running never erases a reviewer's work: non-empty review_notes already in golden.jsonl are carried over by id.
"""
import json
from pathlib import Path
from pipeline.store import read_stage, write_stage

_RANK = {"Error": 0, "Warning": 1, "Info": 2}


def _existing_notes(out_dir):
    try:
        return {rid: r["review_notes"] for rid, r in read_stage(out_dir, "golden").items() if r.get("review_notes")}
    except FileNotFoundError:
        return {}


def write_final(records, out_dir):
    """Write both files into out_dir (created if missing); returns {'golden': path, 'flags_report': path}."""
    out_dir = Path(out_dir)
    notes = _existing_notes(out_dir)
    rows = [{**r, "review_notes": r.get("review_notes") or notes.get(r["id"], "")} for r in records]
    golden = write_stage(out_dir, "golden", rows)

    flagged, by_severity, by_code = [], {}, {}
    for r in rows:
        if not r["flags"]:
            continue
        for f in r["flags"]:
            by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1
            by_code[f["code"]] = by_code.get(f["code"], 0) + 1
        flagged.append({"id": r["id"], "claim_id": r["golden"]["header"]["claim_id"],
                        "worst": min((f["severity"] for f in r["flags"]), key=_RANK.get), "flags": r["flags"]})
    flagged.sort(key=lambda r: (_RANK[r["worst"]], r["id"]))
    report = {"summary": {"records": len(rows), "records_with_flags": len(flagged),
                          "by_severity": by_severity, "by_code": by_code}, "records": flagged}
    path = out_dir / "flags_report.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return {"golden": golden, "flags_report": path}
