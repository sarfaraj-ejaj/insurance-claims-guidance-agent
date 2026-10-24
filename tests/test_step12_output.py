import json
import tempfile
import unittest
from pathlib import Path
from pipeline.step11_validate import assemble
from pipeline.step12_output import write_final
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT
from tests.test_step11_validate import stages_for

SHAPE = ["flags", "golden", "id", "raw_text", "review_notes", "version"]  # sorted: files use sorted keys


def build(n=30):
    out = []
    for i, r in enumerate(RECORDS[:n]):
        d = r["ground_truth"]["incident_description"]
        st = stages_for(r, d["incident_type"], d["location_type"], d["estimated_damage_amount"])
        out.append(assemble(f"rec-{i + 1:03d}", r["claim_text"], st))
    return out


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


class Step12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = build()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "final"  # does not exist yet: write_final must create it

    def tearDown(self):
        self.tmp.cleanup()

    def test_golden_jsonl_has_one_adr_shaped_line_per_record(self):
        paths = write_final(self.records, self.out)
        rows = lines(paths["golden"])
        self.assertEqual(paths["golden"].name, "golden.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertEqual([r["id"] for r in rows], [f"rec-{i:03d}" for i in range(1, 31)])
        for row, src in zip(rows, self.records):
            self.assertEqual(sorted(row), SHAPE)
            self.assertEqual(row["raw_text"], src["raw_text"])  # untouched, byte for byte
            self.assertEqual(row["golden"], src["golden"])

    def test_ac8_flags_report_lists_exactly_the_records_that_have_flags(self):
        paths = write_final(self.records, self.out)
        report = json.loads(paths["flags_report"].read_text(encoding="utf-8"))
        flagged = {r["id"] for r in self.records if r["flags"]}
        self.assertEqual({r["id"] for r in report["records"]}, flagged)
        self.assertEqual(report["summary"]["records"], 30)
        self.assertEqual(report["summary"]["records_with_flags"], len(flagged))
        by_code = {}
        for r in self.records:
            for f in r["flags"]:
                by_code[f["code"]] = by_code.get(f["code"], 0) + 1
        self.assertEqual(report["summary"]["by_code"], by_code)
        self.assertEqual(sum(report["summary"]["by_severity"].values()), sum(by_code.values()))

    def test_flags_report_puts_the_records_needing_review_first(self):
        paths = write_final(self.records, self.out)
        report = json.loads(paths["flags_report"].read_text(encoding="utf-8"))
        rank = {"Error": 0, "Warning": 1, "Info": 2}
        worst = [rank[r["worst"]] for r in report["records"]]
        self.assertEqual(worst, sorted(worst))
        for r in report["records"]:
            self.assertEqual(r["worst"], min((f["severity"] for f in r["flags"]), key=rank.get))
            self.assertIn("claim_id", r)

    def test_rewrite_is_byte_identical(self):
        first = {k: p.read_bytes() for k, p in write_final(self.records, self.out).items()}
        again = {k: p.read_bytes() for k, p in write_final(self.records, self.out).items()}
        self.assertEqual(first, again)

    def test_reviewer_notes_survive_a_rerun(self):
        paths = write_final(self.records, self.out)
        rows = lines(paths["golden"])
        rows[3]["review_notes"] = "checked with the policy file: expiry is right"
        paths["golden"].write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
        write_final(self.records, self.out)  # the pipeline runs again and produces empty notes
        again = lines(paths["golden"])
        self.assertEqual(again[3]["review_notes"], "checked with the policy file: expiry is right")
        self.assertEqual(again[4]["review_notes"], "")

    def test_empty_input_and_non_ascii_text(self):
        paths = write_final([], self.out)
        self.assertEqual(paths["golden"].read_text(encoding="utf-8"), "")
        report = json.loads(paths["flags_report"].read_text(encoding="utf-8"))
        self.assertEqual(report["summary"], {"records": 0, "records_with_flags": 0, "by_severity": {}, "by_code": {}})
        rec = assemble("rec-001", "I’m José — café", {})
        write_final([rec], self.out)
        self.assertEqual(lines(paths["golden"])[0]["raw_text"], "I’m José — café")

    def test_files_are_written_only_inside_the_output_folder(self):
        write_final(self.records, self.out)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["flags_report.json", "golden.jsonl"])


if __name__ == "__main__":
    unittest.main()
