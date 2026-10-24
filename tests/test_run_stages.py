import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from pipeline import run_stages
from pipeline.step01_ingest import ingest as ingest_fn
from pipeline.step02_header import extract_header
from pipeline.store import read_stage, write_stage
from tests.test_step01_ingest import RECORDS, ROOT

RAW = ROOT / "insurance_claims.json"
EXPECTED = ["01_ingest", "02_header", "03_claim_id", "04_policy", "05_policy_dates",
            "06_objects", "07_incident_date", "10_police"]


def ingest_text(raw):
    return ingest_fn(raw)["text"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Store(unittest.TestCase):
    def test_write_read_roundtrip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "rec-002", "b": 1, "a": "é"}, {"id": "rec-001", "x": None}]
            p = write_stage(d, "demo", rows)
            first = p.read_bytes()
            self.assertEqual(first, (b'{"a":"\xc3\xa9","b":1,"id":"rec-002"}\n{"id":"rec-001","x":null}\n'))
            self.assertEqual(read_stage(d, "demo")["rec-001"], {"id": "rec-001", "x": None})
            write_stage(d, "demo", rows)
            self.assertEqual(p.read_bytes(), first)  # byte-identical rewrite

    def test_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                read_stage(d, "nope")
            with self.assertRaises(ValueError):
                write_stage(d, "dup", [{"id": "a"}, {"id": "a"}])
            with self.assertRaises(ValueError):
                write_stage(d, "noid", [{"x": 1}])


class Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.raw_before = sha(RAW)
        cls.paths = run_stages.run(RAW, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_file_per_finished_step_with_30_rows(self):
        self.assertEqual(list(self.paths), EXPECTED)
        for name in EXPECTED:
            rows = read_stage(self.out, name)
            self.assertEqual(len(rows), 30, name)
            self.assertEqual(sorted(rows)[0], "rec-001")
            self.assertEqual(sorted(rows)[-1], "rec-030")

    def test_raw_file_is_never_modified(self):
        self.assertEqual(sha(RAW), self.raw_before)

    def test_no_ground_truth_leaks_into_stage_files(self):
        for path in self.paths.values():
            self.assertNotIn("ground_truth", path.read_text(encoding="utf-8"))

    def test_stage_content_matches_the_step_functions(self):
        ingest = read_stage(self.out, "01_ingest")
        header = read_stage(self.out, "02_header")
        for i, r in enumerate(RECORDS):
            rid = f"rec-{i + 1:03d}"
            self.assertEqual(ingest[rid]["text"], ingest_text(r["claim_text"]))
            want = extract_header(ingest[rid]["text"])
            self.assertEqual({k: header[rid][k] for k in want}, want)
        self.assertTrue(ingest["rec-001"]["truncated"])  # record 0 ends mid-sentence

    def test_records_without_a_policy_number_have_null_policy_layers(self):
        policy = read_stage(self.out, "04_policy")
        dates = read_stage(self.out, "05_policy_dates")
        objs = read_stage(self.out, "06_objects")
        none_ids = [f"rec-{i + 1:03d}" for i in (2, 8, 10, 11, 25)]  # records 2, 8, 10, 11, 25 (0-based)
        for rid in none_ids:
            self.assertIsNone(policy[rid]["policy"], rid)
            self.assertIsNone(dates[rid]["policy_dates"], rid)
            self.assertIsNone(objs[rid]["insured_objects"], rid)
        self.assertEqual(policy["rec-001"]["policy"]["policy_number"], "POL-787532354")

    def test_rerun_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            again = run_stages.run(RAW, d)
            for name in EXPECTED:
                self.assertEqual(sha(again[name]), sha(self.paths[name]), name)

    def test_later_stages_read_earlier_stage_files(self):
        # corrupt the 01 file on disk: stage 03 must follow the file, proving the data really flows through disk
        with tempfile.TemporaryDirectory() as d:
            run_stages.run(RAW, d, only=["01_ingest"])
            rows = read_stage(d, "01_ingest")
            rows["rec-001"]["text"] = "Claim ref CLM-111111"
            write_stage(d, "01_ingest", list(rows.values()))
            run_stages.run(RAW, d, only=["02_header", "03_claim_id"])
            self.assertEqual(read_stage(d, "03_claim_id")["rec-001"]["claim_id"], "CLM-111111")


if __name__ == "__main__":
    unittest.main()
