import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from pipeline import llm, run_stages
from pipeline.store import read_stage
from tests.test_llm import ok
from tests.test_step01_ingest import RECORDS, ROOT

RAW = ROOT / "insurance_claims.json"
ALL = ["01_ingest", "02_header", "03_claim_id", "04_policy", "05_policy_dates", "06_objects",
       "07_incident_date", "08_types", "09_damage", "10_police"]


class Router:
    """Fake Claude: answers the step-8 prompt with labels and the step-9 prompt with no estimate."""

    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if "damage amount" in kw["system"]:
            return ok(json.dumps({"estimated_damage_amount": None, "evidence_quote": None}))
        return ok(json.dumps({"incident_type": "storm_damage", "location_type": "residence_exterior"}))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class LlmStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out, self.cache = Path(self.tmp.name) / "stages", Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def test_llm_stages_write_their_own_files_with_model_and_prompt_version(self):
        c = Router()
        paths = run_stages.run(RAW, self.out, use_llm=True, client=c, cache_dir=self.cache)
        self.assertEqual(list(paths), ALL)
        for name in ("08_types", "09_damage"):
            rows = read_stage(self.out, name)
            self.assertEqual(len(rows), 30)
            row = rows["rec-001"]
            self.assertEqual(row["model"], llm.MODEL)
            self.assertTrue(row["prompt_version"].startswith("step0"))
        self.assertEqual(read_stage(self.out, "08_types")["rec-005"]["incident_type"], "storm_damage")
        self.assertEqual(read_stage(self.out, "09_damage")["rec-005"]["estimated_damage_amount"], None)
        self.assertEqual(len(c.calls), 60)  # 30 claims x 2 LLM steps

    def test_the_prompt_carries_the_ingested_claim_text(self):
        c = Router()
        run_stages.run(RAW, self.out, only=["01_ingest", "08_types"], use_llm=True, client=c, cache_dir=self.cache)
        sent = [call["messages"][0]["content"] for call in c.calls]
        text = read_stage(self.out, "01_ingest")["rec-001"]["text"]
        self.assertTrue(any(text in s for s in sent))
        self.assertEqual(len(c.calls), 30)

    def test_second_run_is_answered_from_the_cache(self):
        c = Router()
        run_stages.run(RAW, self.out, use_llm=True, client=c, cache_dir=self.cache)
        first = {n: sha(self.out / f"{n}.jsonl") for n in ALL}
        run_stages.run(RAW, self.out, use_llm=True, client=c, cache_dir=self.cache)
        self.assertEqual(len(c.calls), 60)  # no new model calls
        self.assertEqual({n: sha(self.out / f"{n}.jsonl") for n in ALL}, first)  # byte-identical

    def test_without_credentials_the_stage_still_completes_with_flagged_nulls(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            run_stages.run(RAW, self.out, use_llm=True, cache_dir=self.cache)
        for name in ("08_types", "09_damage"):
            for row in read_stage(self.out, name).values():
                self.assertEqual(row["flags"], ["llm_unavailable"])
        row = read_stage(self.out, "08_types")["rec-001"]
        self.assertEqual((row["incident_type"], row["location_type"]), (None, None))
        self.assertEqual(list(self.cache.glob("*.json")) if self.cache.exists() else [], [])

    def test_default_run_is_unchanged_and_makes_no_model_calls(self):
        paths = run_stages.run(RAW, self.out)
        self.assertEqual(list(paths), [n for n in ALL if n not in ("08_types", "09_damage")])
        self.assertFalse(self.cache.exists())

    def test_secrets_never_reach_the_stage_files(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-SECRET"}):
            run_stages.run(RAW, self.out, use_llm=True, client=Router(), cache_dir=self.cache)
        for p in list(self.out.glob("*.jsonl")) + list(self.cache.glob("*.json")):
            self.assertNotIn("SECRET", p.read_text(encoding="utf-8"))

    def test_raw_file_is_still_never_modified(self):
        before = sha(RAW)
        run_stages.run(RAW, self.out, use_llm=True, client=Router(), cache_dir=self.cache)
        self.assertEqual(sha(RAW), before)


if __name__ == "__main__":
    unittest.main()
