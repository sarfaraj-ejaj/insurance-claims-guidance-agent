"""Fine-tuning dataset preparation, end to end (FR-15..FR-20).

    py -m pipeline.finetune [--raw insurance_claims.json] [--out data]

Needs data/final/golden.jsonl (made by `py -m pipeline`; if it is missing it is made first). Then:
    audit -> reconcile -> de-identify -> export -> split (80/20) -> independent checks
and writes, under --out:
    audit/     field_agreement.json  conflicts.jsonl  profile.json  deid_report.json  readiness_report.json
    review/    reconciled.jsonl  excluded.jsonl  decisions.jsonl (yours)      <- contain ORIGINAL values, keep private
    finetune/  deidentified.jsonl  examples.jsonl  train.jsonl  validation.jsonl  split_manifest.json  ...  <- masked only
Exit code: 0 when the checks pass (with or without warnings), 1 when a check fails, 2 when it cannot run.
"""
import argparse
import json
import sys
from pathlib import Path
from pipeline import __main__ as extraction
from pipeline import ft_audit, ft_check, ft_deid, ft_export, ft_reconcile, ft_split
from pipeline.store import read_stage

ROOT = Path(__file__).resolve().parent.parent


def _rows(directory, name):
    return sorted(read_stage(directory, name).values(), key=lambda r: r["id"])


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(prog="pipeline.finetune", description="Golden records -> fine-tuning train/validation files.")
    ap.add_argument("--raw", default=str(ROOT / "insurance_claims.json"), help="raw claims JSON (read only)")
    ap.add_argument("--out", default=str(ROOT / "data"), help="data folder holding final/, audit/, review/, finetune/")
    args = ap.parse_args(argv)
    raw, out = Path(args.raw), Path(args.out)
    golden = out / "final" / "golden.jsonl"
    review, ft, audit = out / "review", out / "finetune", out / "audit"

    if not golden.exists():
        print("No golden records yet - running the extraction pipeline first.")
        extraction.main(["--raw", str(raw), "--out", str(out)], client=client)

    a = ft_audit.run(golden, raw, audit)
    summary = json.loads(a["field_agreement"].read_text(encoding="utf-8"))
    print(f"1/6 audit        {summary['records']} records: {summary['conflicts']} conflicting, "
          f"{summary['pipeline_null']} missing from the pipeline output  -> {audit}")

    rec = ft_reconcile.run(golden, raw, review / "decisions.jsonl", review)
    ready, excluded = _rows(review, "reconciled"), _rows(review, "excluded")
    print(f"2/6 reconcile    {len(ready)} ready, {len(excluded)} excluded because the two label sources conflict  -> {review}")
    for e in excluded:
        print(f"                 {e['id']}: " + ", ".join(f"{x['field']} ({x['kind']})" for x in e["reasons"]))
    if excluded:
        print(f"                 to bring one back, add a line to {review / 'decisions.jsonl'}:\n"
              '                 {"record": "rec-008", "field": "policy_details.coverage_type", "choose": "dataset", '
              '"by": "you", "reason": "checked the claim"}')

    _, deid_report = ft_deid.run(rec["reconciled"], ft)
    (audit / "deid_report.json").write_text(json.dumps(deid_report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8", newline="\n")
    leaks = len(deid_report["leaks"]) + len(deid_report["mask_induced_untraceable"])
    print(f"3/6 de-identify  {sum(deid_report['replacements'].values())} replacements, {leaks} leftover/broken link(s), "
          f"{len(deid_report['residual_candidates'])} residual candidate(s) to glance at  -> {ft}")

    _, rejected = ft_export.write_examples(_rows(ft, "deidentified"), ft)
    print(f"4/6 export       {len(_rows(ft, 'deidentified')) - len(rejected)} examples, {len(rejected)} rejected by the schema check")
    try:
        ft_split.write_split(ft)
    except ValueError as e:
        print(f"5/6 split        cannot split: {e}")
        return 2
    manifest = json.loads((ft / "split_manifest.json").read_text(encoding="utf-8"))
    c = manifest["counts"]
    print(f"5/6 split        train {c['train']} / validation {c['validation']} (80/20, no test set)")

    path, report = ft_check.run(out)
    print(f"6/6 checks       verdict: {report['verdict'].upper()}  -> {path}")
    for check in report["checks"]:
        print(f"                 [{check['status']:<4}] {check['name']:<13} {check['detail'][:110]}")
    for w in report["warnings"]:
        print(f"                 ! {w}")
    return 1 if report["verdict"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
