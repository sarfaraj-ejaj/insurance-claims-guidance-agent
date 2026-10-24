import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from pipeline.__main__ import main
from tests.test_llm import ok
from tests.test_run_stages_llm import Router
from tests.test_step01_ingest import REFERENCE_TEXT, ROOT
from tests.test_step11_validate import ADR_GOLDEN

RAW = ROOT / "insurance_claims.json"


def sha_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def run_main(*argv, client=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(argv), client=client)
    return code, buf.getvalue()


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


class ReferenceClient:
    """Fake Claude that answers like the ADR section 8 reference record."""

    def __init__(self):
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        if "damage amount" in kw["system"]:
            return ok(json.dumps({"estimated_damage_amount": 11960,
                                  "evidence_quote": "the written quote is\n$11,960"}))
        return ok(json.dumps({"incident_type": "storm_damage", "location_type": "residence_exterior"}))


class Main(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "data"

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_writes_stages_final_derived_and_cache_and_never_touches_the_raw_file(self):
        before = hashlib.sha256(RAW.read_bytes()).hexdigest()
        code, text = run_main("--out", str(self.out), client=Router())
        self.assertEqual(code, 0)
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(), before)
        self.assertEqual(len(list((self.out / "stages").glob("*.jsonl"))), 10)
        self.assertEqual(sorted(p.name for p in (self.out / "final").iterdir()),
                         ["derived.jsonl", "flags_report.json", "golden.jsonl"])
        self.assertTrue(list((self.out / "llm_cache").glob("*.json")))
        rows = lines(self.out / "final" / "golden.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertEqual(len(lines(self.out / "final" / "derived.jsonl")), 30)
        self.assertIn("30 records", text)

    def test_ac1_the_reference_record_comes_out_as_the_adr_golden_output(self):
        raw = Path(self.tmp.name) / "ref.json"
        raw.write_text(json.dumps([{"claim_text": REFERENCE_TEXT, "ground_truth": {}}]), encoding="utf-8")
        run_main("--raw", str(raw), "--out", str(self.out), client=ReferenceClient())
        (rec,) = lines(self.out / "final" / "golden.jsonl")
        self.assertEqual(rec["golden"], ADR_GOLDEN)
        self.assertEqual([f for f in rec["flags"] if f["severity"] == "Error"], [])

    def test_ac2_ac8_no_error_flags_when_the_llm_steps_answer(self):
        run_main("--out", str(self.out), client=Router())
        for rec in lines(self.out / "final" / "golden.jsonl"):
            self.assertEqual([f["code"] for f in rec["flags"] if f["severity"] == "Error"], [], rec["id"])
        report = json.loads((self.out / "final" / "flags_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["records"], 30)

    def test_ac6_a_second_run_is_byte_identical(self):
        run_main("--out", str(self.out), client=Router())
        first = sha_tree(self.out)
        run_main("--out", str(self.out), client=Router())
        self.assertEqual(sha_tree(self.out), first)

    def test_without_an_api_key_it_finishes_with_flagged_nulls_and_says_how_to_fix_it(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            code, text = run_main("--out", str(self.out))
        self.assertEqual(code, 0)
        rows = lines(self.out / "final" / "golden.jsonl")
        for rec in rows:
            g = rec["golden"]["incident_description"]
            self.assertEqual((g["incident_type"], g["location_type"], g["estimated_damage_amount"]), (None,) * 3)
            self.assertIn("llm_unavailable", [f["code"] for f in rec["flags"]])
        self.assertEqual(rows[0]["golden"]["header"]["claim_id"], "CLM-424063")  # rule-based layers are complete
        self.assertIn("ANTHROPIC_API_KEY", text)
        self.assertEqual(list((self.out / "llm_cache").glob("*.json")) if (self.out / "llm_cache").exists() else [], [])

    def test_no_llm_flag_skips_the_model_and_marks_the_missing_stages(self):
        code, _ = run_main("--out", str(self.out), "--no-llm")
        self.assertEqual(code, 0)
        self.assertFalse((self.out / "stages" / "08_types.jsonl").exists())
        rec = lines(self.out / "final" / "golden.jsonl")[0]
        self.assertEqual(sum(f["code"] == "stage_missing" for f in rec["flags"]), 2)

    def test_reviewer_notes_survive_a_rerun(self):
        run_main("--out", str(self.out), client=Router())
        path = self.out / "final" / "golden.jsonl"
        rows = lines(path)
        rows[1]["review_notes"] = "looked at the original email; fine"
        path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
        run_main("--out", str(self.out), client=Router())
        self.assertEqual(lines(path)[1]["review_notes"], "looked at the original email; fine")

    def test_it_really_runs_as_py_dash_m_pipeline(self):
        done = subprocess.run([sys.executable, "-m", "pipeline", "--no-llm", "--out", str(self.out)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((self.out / "final" / "golden.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
