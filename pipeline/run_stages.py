"""Run the finished pipeline steps over the raw claims, one stage file per step.

    py -m pipeline.run_stages [raw.json] [out_dir] [--llm]   (defaults: insurance_claims.json -> data/stages)

--llm also runs stages 08 (incident/location type) and 09 (damage amount) through Claude; needs ANTHROPIC_API_KEY.

Each stage reads the earlier stage files from disk and writes only its own fields. The raw file is never touched.
"""
import json
import sys
from pathlib import Path
from pipeline import llm
from pipeline.step01_ingest import ingest
from pipeline.step02_header import extract_header
from pipeline.step03_claim_id import extract_claim_id
from pipeline.step04_policy import extract_policy
from pipeline.step05_policy_dates import extract_policy_dates
from pipeline.step06_objects import extract_objects
from pipeline.step07_incident_date import extract_incident_date
from pipeline.step08_types import classify
from pipeline.step09_damage import estimate_damage
from pipeline.step10_police import extract_police_report_number
from pipeline.store import read_stage, write_stage

ROOT = Path(__file__).resolve().parent.parent


def _s01(rec, prev):
    out = ingest(rec["claim_text"])
    return {"text": out["text"], "truncated": out["truncated"]}


def _s02(rec, prev):
    return extract_header(prev("01_ingest")["text"])


def _s03(rec, prev):
    return {"claim_id": extract_claim_id(prev("01_ingest")["text"])}


def _s04(rec, prev):
    return {"policy": extract_policy(prev("01_ingest")["text"], prev("02_header")["reported_by"])}


def _s05(rec, prev):
    if prev("04_policy")["policy"] is None:  # no policy number -> no policy layers (ADR section 2)
        return {"policy_dates": None}
    return {"policy_dates": extract_policy_dates(prev("01_ingest")["text"], prev("02_header")["report_date"])}


def _s06(rec, prev):
    pol = prev("04_policy")["policy"]
    return {"insured_objects": extract_objects(prev("01_ingest")["text"], pol["coverage_type"]) if pol else None}


def _s07(rec, prev):
    d, flags = extract_incident_date(prev("01_ingest")["text"], prev("02_header")["report_date"])
    return {"incident_date": d, "flags": flags}


def _s10(rec, prev):
    return {"police_report_number": extract_police_report_number(prev("01_ingest")["text"])}


# stage name -> function(raw_record, prev) -> that step's fields (rule-based; no network, no key needed)
STAGES = {"01_ingest": _s01, "02_header": _s02, "03_claim_id": _s03, "04_policy": _s04,
          "05_policy_dates": _s05, "06_objects": _s06, "07_incident_date": _s07, "10_police": _s10}


def llm_stages(client=None, cache_dir=llm.CACHE_DIR):
    """The two Claude-backed stages (opt-in). Without a key or on any failure a row is null plus a flag."""
    return {
        "08_types": lambda rec, prev: classify(prev("01_ingest")["text"], client=client, cache_dir=cache_dir),
        "09_damage": lambda rec, prev: estimate_damage(prev("01_ingest")["text"], client=client, cache_dir=cache_dir)}


def run(raw_path=ROOT / "insurance_claims.json", out_dir=ROOT / "data" / "stages", only=None, stages=None,
        use_llm=False, client=None, cache_dir=llm.CACHE_DIR):
    """Run the stages (all, or just those named in `only`) and return {stage name: file path}.

    use_llm=True adds stages 08 and 09; `client` and `cache_dir` are for tests and the cache location.
    """
    stages = stages or ({**STAGES, **llm_stages(client, cache_dir)} if use_llm else STAGES)
    names = [n for n in sorted(stages) if only is None or n in only]
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    ids = [f"rec-{i + 1:03d}" for i in range(len(raw))]
    cache = {}

    def stage_rows(name):  # earlier stages are always read back from their files
        if name not in cache:
            cache[name] = read_stage(out_dir, name)
        return cache[name]

    paths = {}
    for name in names:
        rows = []
        for rid, rec in zip(ids, raw):
            prev = lambda s, rid=rid: stage_rows(s)[rid]
            rows.append({"id": rid, **stages[name](rec, prev)})
        paths[name] = write_stage(out_dir, name, rows)
        cache.pop(name, None)  # force readers to see what is on disk
    return paths


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--llm"]
    done = run(*(Path(a) for a in args[:2]), use_llm="--llm" in sys.argv)
    for name, path in done.items():
        print(f"{name:<18} {path}")
