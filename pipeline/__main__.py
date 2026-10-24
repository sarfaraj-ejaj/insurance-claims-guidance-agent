"""End to end: raw claims -> data/stages (one file per step) -> data/final (golden, flags, derived).

    py -m pipeline [--raw insurance_claims.json] [--out data] [--no-llm]

The raw file is never modified. Steps 8 and 9 use Claude (set ANTHROPIC_API_KEY, `pip install anthropic`); without
a key they are left null with an llm_unavailable flag and everything else still runs. Reruns are byte-identical
and keep reviewers' review_notes.
"""
import argparse
import json
import sys
from pathlib import Path
from pipeline import run_stages
from pipeline.step11_validate import assemble
from pipeline.step12_output import write_final
from pipeline.step13_derived import write_derived
from pipeline.store import read_stage

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(prog="pipeline", description="Raw insurance claims -> golden records.")
    ap.add_argument("--raw", default=str(ROOT / "insurance_claims.json"), help="raw claims JSON (never modified)")
    ap.add_argument("--out", default=str(ROOT / "data"), help="output folder: stages/, final/, llm_cache/")
    ap.add_argument("--no-llm", action="store_true", help="skip the Claude steps (8 and 9)")
    args = ap.parse_args(argv)
    out = Path(args.out)

    run_stages.run(args.raw, out / "stages", use_llm=not args.no_llm, client=client, cache_dir=out / "llm_cache")
    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    stage_files = sorted(p.stem for p in (out / "stages").glob("*.jsonl"))
    rows = {name: read_stage(out / "stages", name) for name in stage_files}
    records = [assemble(f"rec-{i + 1:03d}", r["claim_text"], {n: rows[n][f"rec-{i + 1:03d}"] for n in rows})
               for i, r in enumerate(raw)]
    write_final(records, out / "final")
    write_derived(records, out / "final")

    sev = {}
    for r in records:
        for f in r["flags"]:
            sev[f["severity"]] = sev.get(f["severity"], 0) + 1
    unavailable = sum(f["code"] == "llm_unavailable" for r in records for f in r["flags"])
    print(f"{len(records)} records -> {out / 'final' / 'golden.jsonl'}")
    print(f"stages: {len(stage_files)} files in {out / 'stages'}")
    print("flags: " + (", ".join(f"{k} {sev[k]}" for k in ("Error", "Warning", "Info") if k in sev) or "none"))
    if unavailable:
        print("Claude steps were skipped (no usable answer): set ANTHROPIC_API_KEY, run `pip install anthropic`, "
              "then rerun this command to fill incident type, location type and damage amount.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
