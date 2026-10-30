import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from pipeline import ft_check as fc, ft_deid, ft_export, ft_split
from pipeline.finetune import main as finetune_main
from tests.test_ft_audit import golden_rows
from tests.test_ft_deid import dataset_rows, leaks as orig_leaks, originals
from tests.test_step01_ingest import RECORDS


def build(tmp, rows=None):
    """A complete, valid fine-tuning folder built through the real modules; returns (dir, reconciled rows)."""
    rows = rows or dataset_rows()
    masked, _ = ft_deid.deidentify(rows)
    d = Path(tmp) / "finetune"
    ft_export.write_examples(masked, d)
    ft_split.write_split(d)
    return d, rows


def lines(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(p, items):
    Path(p).write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")


def status(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


class Good(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir, cls.rows = build(cls.tmp.name)
        cls.report = fc.run_checks(cls.dir, cls.rows, [])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_valid_export_passes_every_check_and_warns_about_size(self):
        self.assertEqual([c["name"] for c in self.report["checks"]], fc.CHECKS)
        failing = [c for c in self.report["checks"] if c["status"] == "fail"]
        self.assertEqual(failing, [], failing)
        self.assertEqual(self.report["verdict"], "pass_with_warnings")
        self.assertEqual(status(self.report, "size"), "warn")

    def test_the_report_states_the_dataset_size_and_a_plain_warning(self):
        ds = self.report["dataset"]
        self.assertEqual((ds["examples"], ds["train"], ds["validation"]), (30, 24, 6))
        warning = " ".join(self.report["warnings"])
        self.assertIn("24", warning)
        self.assertIn("too small", warning)
        self.assertIn("evaluation set", warning)

    def test_every_check_has_a_readable_detail(self):
        for c in self.report["checks"]:
            self.assertTrue(c["detail"].strip(), c["name"])

    def test_the_report_contains_no_original_personal_value(self):
        text = json.dumps(self.report, ensure_ascii=False)
        for r in self.rows:
            for n in r["target"]["header"]["reported_by"].split():
                self.assertNotIn(n, text)


class Bad(unittest.TestCase):
    """Corrupt a good export in one way at a time: the checker must notice every one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir, self.rows = build(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def verdict(self, excluded=()):
        return fc.run_checks(self.dir, self.rows, list(excluded))

    def assert_fails(self, name, report=None):
        report = report or self.verdict()
        self.assertEqual(report["verdict"], "fail")
        self.assertEqual(status(report, name), "fail", [c for c in report["checks"] if c["status"] == "fail"])

    def test_an_original_name_in_the_training_file_is_a_leak(self):
        orig = self.rows[0]["target"]["header"]["reported_by"].split()[1]
        tr = lines(self.dir / "train.jsonl")
        tr[0]["messages"][1]["content"] += f"\nSigned: {orig}"
        write(self.dir / "train.jsonl", tr)
        self.assert_fails("personal_data")

    def test_an_original_id_in_the_manifest_file_is_a_leak_too(self):
        m = json.loads((self.dir / "split_manifest.json").read_text(encoding="utf-8"))
        m["note"] = self.rows[0]["target"]["policy_details"]["policy_number"]
        (self.dir / "split_manifest.json").write_text(json.dumps(m), encoding="utf-8")
        self.assert_fails("personal_data")

    def test_a_line_present_in_both_train_and_validation_breaks_the_split(self):
        tr, va = lines(self.dir / "train.jsonl"), lines(self.dir / "validation.jsonl")
        va.append(tr[0])
        write(self.dir / "validation.jsonl", va)
        report = self.verdict()
        self.assert_fails("split", report)
        self.assertEqual(status(report, "duplicates"), "fail")

    def test_a_wrong_ratio_is_caught(self):
        tr, va = lines(self.dir / "train.jsonl"), lines(self.dir / "validation.jsonl")
        write(self.dir / "train.jsonl", tr[3:])
        write(self.dir / "validation.jsonl", va + tr[:3])
        self.assert_fails("split")

    def test_an_invalid_label_is_caught_by_the_schema_check(self):
        tr = lines(self.dir / "train.jsonl")
        answer = json.loads(tr[0]["messages"][2]["content"])
        answer["incident_description"]["incident_type"] = "flood"
        tr[0]["messages"][2]["content"] = json.dumps(answer, separators=(",", ":"))
        write(self.dir / "train.jsonl", tr)
        self.assert_fails("schema")

    def test_a_label_that_masking_cut_loose_from_its_text_is_caught(self):
        tr = lines(self.dir / "train.jsonl")
        answer = json.loads(tr[0]["messages"][2]["content"])
        user = tr[0]["messages"][1]["content"]
        name = answer["header"]["reported_by"]
        tr[0]["messages"][1]["content"] = user.replace(name.split()[0], "Someone")
        write(self.dir / "train.jsonl", tr)
        self.assert_fails("traceability")

    def test_an_excluded_record_that_reached_the_export_is_caught(self):
        manifest = json.loads((self.dir / "split_manifest.json").read_text(encoding="utf-8"))
        self.assert_fails("exclusions", self.verdict(excluded=[{"id": manifest["train_ids"][0], "reasons": []}]))

    def test_a_test_set_or_an_internal_id_is_caught(self):
        shutil.copy(self.dir / "validation.jsonl", self.dir / "test.jsonl")
        self.assert_fails("file_shape")
        (self.dir / "test.jsonl").unlink()
        tr = lines(self.dir / "train.jsonl")
        tr[0]["id"] = "rec-001"
        write(self.dir / "train.jsonl", tr)
        self.assert_fails("file_shape")

    def test_a_file_that_a_rerun_would_not_reproduce_is_caught(self):
        tr = lines(self.dir / "train.jsonl")
        write(self.dir / "train.jsonl", list(reversed(tr)))  # same content, different bytes
        self.assert_fails("determinism")

    def test_missing_files_fail_cleanly_instead_of_crashing(self):
        (self.dir / "validation.jsonl").unlink()
        report = self.verdict()
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("validation.jsonl", json.dumps(report))


class Warnings(unittest.TestCase):
    def test_label_values_the_dataset_never_had_in_the_text_are_reported_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = build(tmp)
            report = fc.run_checks(d, rows, [])
            trace = next(c for c in report["checks"] if c["name"] == "traceability")
            self.assertEqual(trace["status"], "pass")
            self.assertGreaterEqual(trace["not_in_text_before_masking"], 3)  # e.g. the inferred holder, address fragments

    def test_residual_candidates_from_the_deid_report_are_passed_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = build(tmp)
            report = fc.run_checks(d, rows, [], deid_report={"residual_candidates": {"Safeway": 1}})
            self.assertEqual(report["residual_candidates"], {"Safeway": 1})
            self.assertTrue(any("Safeway" in w or "residual" in w for w in report["warnings"]))


class Cli(unittest.TestCase):
    """`py -m pipeline.finetune` end to end, in a temp folder, with the Claude steps stood in by dataset labels."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "data"
        (self.out / "final").mkdir(parents=True)
        write(self.out / "final" / "golden.jsonl", golden_rows())
        self.raw = Path(self.tmp.name) / "raw.json"
        self.raw.write_text(json.dumps(RECORDS), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = finetune_main(["--raw", str(self.raw), "--out", str(self.out), *extra])
        return code, buf.getvalue()

    def tree(self):
        return {str(p.relative_to(self.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.out.rglob("*")) if p.is_file()}

    def test_full_run_makes_every_file_and_an_80_20_split_of_the_usable_records(self):
        code, text = self.run_cli()
        self.assertEqual(code, 0)
        for rel in ("audit/field_agreement.json", "audit/conflicts.jsonl", "audit/profile.json", "audit/deid_report.json",
                    "audit/readiness_report.json", "review/reconciled.jsonl", "review/excluded.jsonl",
                    "finetune/deidentified.jsonl", "finetune/examples.jsonl", "finetune/examples_index.json",
                    "finetune/train.jsonl", "finetune/validation.jsonl", "finetune/split_manifest.json"):
            self.assertTrue((self.out / rel).exists(), rel)
        self.assertEqual((len(lines(self.out / "finetune" / "train.jsonl")), len(lines(self.out / "finetune" / "validation.jsonl"))), (21, 5))
        report = json.loads((self.out / "audit" / "readiness_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["verdict"], "pass_with_warnings")
        self.assertIn("4 excluded", text)
        self.assertIn("decisions.jsonl", text)
        self.assertFalse(list((self.out / "finetune").glob("test*")))

    def test_recorded_decisions_bring_the_excluded_records_back_for_a_24_6_split(self):
        (self.out / "review").mkdir()
        decide = lambda r, f: {"record": r, "field": f, "choose": "dataset", "by": "reviewer", "reason": "checked the claim"}
        write(self.out / "review" / "decisions.jsonl", [
            decide("rec-005", "insured_objects[0].year"), decide("rec-008", "policy_details.coverage_type"),
            decide("rec-008", "insured_objects[0].make_model"), decide("rec-008", "insured_objects[0].estimated_value"),
            decide("rec-013", "insured_objects[0].estimated_value"), decide("rec-015", "policy_details.expiration_date")])
        code, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual((len(lines(self.out / "finetune" / "train.jsonl")), len(lines(self.out / "finetune" / "validation.jsonl"))), (24, 6))
        self.assertEqual(lines(self.out / "review" / "excluded.jsonl"), [])

    def test_no_original_personal_value_is_anywhere_under_finetune(self):
        self.run_cli()
        blob = " ".join(p.read_text(encoding="utf-8") for p in (self.out / "finetune").iterdir())
        for r in dataset_rows():
            for kind, value in originals(r["target"]):
                self.assertFalse(orig_leaks(kind, value, blob), kind)

    def test_a_rerun_is_byte_identical_and_inputs_are_never_modified(self):
        golden_before = hashlib.sha256((self.out / "final" / "golden.jsonl").read_bytes()).hexdigest()
        raw_before = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        self.run_cli()
        first = self.tree()
        self.run_cli()
        self.assertEqual(self.tree(), first)
        self.assertEqual(hashlib.sha256((self.out / "final" / "golden.jsonl").read_bytes()).hexdigest(), golden_before)
        self.assertEqual(hashlib.sha256(self.raw.read_bytes()).hexdigest(), raw_before)

    def test_missing_golden_records_are_produced_first(self):
        (self.out / "final" / "golden.jsonl").unlink()
        with mock.patch.dict(os.environ, {}, clear=True):
            code, text = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("running the extraction pipeline first", text)
        self.assertTrue((self.out / "finetune" / "train.jsonl").exists())

    def test_exit_code_is_1_when_a_check_fails_and_2_when_there_is_too_little_to_split(self):
        bad = {"verdict": "fail", "checks": [{"status": "fail", "name": "split", "detail": "x"}], "warnings": [], "dataset": {}}
        with mock.patch("pipeline.finetune.ft_check.run", return_value=(self.out / "audit" / "x.json", bad)):
            self.assertEqual(self.run_cli()[0], 1)
        write(self.out / "final" / "golden.jsonl", golden_rows()[:3])
        self.raw.write_text(json.dumps(RECORDS[:3]), encoding="utf-8")
        code, text = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("too few", text)


if __name__ == "__main__":
    unittest.main()
