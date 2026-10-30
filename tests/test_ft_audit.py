import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from pipeline import ft_audit
from pipeline.store import read_stage
from tests.test_step01_ingest import RECORDS, ROOT
from tests.test_step12_output import build

GOLDEN = None


def golden_rows():
    """Pipeline records with ground-truth labels standing in for the two Claude steps."""
    global GOLDEN
    if GOLDEN is None:
        GOLDEN = build()
    return copy.deepcopy(GOLDEN)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Compare(unittest.TestCase):
    def test_known_conflicts_between_pipeline_and_dataset_are_found_exactly(self):
        _, conflicts = ft_audit.compare(golden_rows(), RECORDS)
        got = sorted((c["record"], c["field"], c["kind"]) for c in conflicts)
        self.assertEqual(got, [
            ("rec-005", "insured_objects[0].year", "pipeline_null"),
            ("rec-008", "insured_objects[0].estimated_value", "conflict"),
            ("rec-008", "insured_objects[0].make_model", "conflict"),  # follows from the wrong coverage below
            ("rec-008", "policy_details.coverage_type", "conflict"),
            ("rec-013", "insured_objects[0].estimated_value", "conflict"),
            ("rec-015", "policy_details.expiration_date", "conflict")])
        c = next(c for c in conflicts if c["field"] == "policy_details.coverage_type")
        self.assertEqual((c["pipeline"], c["dataset"], c["id"]), ("Property", "Liability",
                                                                  "rec-008:policy_details.coverage_type"))

    def test_agreement_counts_per_field(self):
        agree, _ = ft_audit.compare(golden_rows(), RECORDS)
        self.assertEqual(agree["header.claim_id"], {"agree": 30, "total": 30})
        self.assertEqual(agree["policy_details.coverage_type"], {"agree": 29, "total": 30})
        self.assertEqual(agree["policy_details.expiration_date"], {"agree": 29, "total": 30})
        self.assertEqual(agree["incident_description.incident_type"], {"agree": 30, "total": 30})
        self.assertTrue(all(v["total"] == 30 for k, v in agree.items() if not k.startswith("insured_objects")))

    def test_the_real_pipeline_output_without_claude_shows_pipeline_null_not_conflict(self):
        rows = golden_rows()
        for r in rows:
            r["golden"]["incident_description"].update(incident_type=None, location_type=None)
        agree, conflicts = ft_audit.compare(rows, RECORDS)
        self.assertEqual(agree["incident_description.incident_type"], {"agree": 0, "total": 30})
        kinds = {c["kind"] for c in conflicts if c["field"] == "incident_description.incident_type"}
        self.assertEqual(kinds, {"pipeline_null"})

    def test_kinds_on_synthetic_records(self):
        def pair(pipe, data):
            g = {"header": {"claim_id": pipe}, "policy_details": None, "insured_objects": None,
                 "incident_description": {}}
            t = {"header": {"claim_id": data}, "policy_details": None, "insured_objects": None,
                 "incident_description": {}}
            _, conflicts = ft_audit.compare([{"id": "rec-001", "golden": g}], [{"ground_truth": t}])
            return [c["kind"] for c in conflicts]
        self.assertEqual(pair("CLM-111111", "CLM-111111"), [])
        self.assertEqual(pair(None, None), [])
        self.assertEqual(pair(None, "CLM-111111"), ["pipeline_null"])
        self.assertEqual(pair("CLM-111111", None), ["dataset_null"])
        self.assertEqual(pair("CLM-111111", "CLM-222222"), ["conflict"])

    def test_object_count_difference_is_a_conflict_on_the_list(self):
        rows = golden_rows()
        rows[3]["golden"]["insured_objects"] = rows[3]["golden"]["insured_objects"][:1]  # record has 2 vehicles
        _, conflicts = ft_audit.compare(rows, RECORDS)
        self.assertIn(("rec-004", "insured_objects", "conflict"),
                      [(c["record"], c["field"], c["kind"]) for c in conflicts])

    def test_the_inputs_are_not_modified(self):
        rows, raw = golden_rows(), copy.deepcopy(RECORDS)
        before = (copy.deepcopy(rows), copy.deepcopy(raw))
        ft_audit.compare(rows, raw)
        self.assertEqual((rows, raw), before)


class Profile(unittest.TestCase):
    def test_class_balance_and_rare_types(self):
        p = ft_audit.profile(RECORDS)
        self.assertEqual(p["records"], 30)
        it = p["label_balance"]["incident_type"]
        self.assertEqual((len(it), it["slip_and_fall"], it["rear_end_collision"], it["roof_leak"]), (14, 5, 5, 1))
        self.assertEqual(sorted(p["rare_incident_types"]), sorted(k for k, v in it.items() if v <= 2))
        self.assertEqual(len(p["rare_incident_types"]), 10)  # SPEC risk text says 7: it is 10 of 14
        self.assertEqual(p["label_balance"]["coverage_type"]["none"], 5)
        self.assertEqual(p["duplicate_texts"], 0)

    def test_personal_data_inventory(self):
        pd = ft_audit.profile(RECORDS)["personal_data"]
        self.assertEqual(pd["labels"]["distinct_person_names"], 31)
        self.assertEqual(pd["labels"]["vins"], 10)
        self.assertEqual(pd["labels"]["policy_numbers"], 25)
        self.assertEqual(pd["labels"]["claim_ids"], 29)
        self.assertEqual(pd["labels"]["police_report_numbers"], 14)
        self.assertEqual(pd["labels"]["street_addresses"], 15)
        self.assertEqual(pd["labels"]["repeated_first_names"]["Jennifer"], 4)
        self.assertEqual(pd["text_scan"], {"email": 0, "phone": 0, "ssn_like": 0})

    def test_text_length_stats(self):
        t = ft_audit.profile(RECORDS)["text_chars"]
        self.assertEqual((t["min"], t["max"]), (487, 2744))
        self.assertTrue(t["min"] < t["avg"] < t["max"])


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.golden = self.dir / "golden.jsonl"
        self.golden.write_text("\n".join(json.dumps(r, sort_keys=True) for r in golden_rows()) + "\n", encoding="utf-8")
        self.raw = self.dir / "raw.json"
        self.raw.write_text(json.dumps(RECORDS), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_three_files_deterministically_and_leaves_inputs_alone(self):
        before = (sha(self.golden), sha(self.raw))
        out = self.dir / "audit"
        paths = ft_audit.run(self.golden, self.raw, out)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["conflicts.jsonl", "field_agreement.json", "profile.json"])
        first = {k: sha(p) for k, p in paths.items()}
        ft_audit.run(self.golden, self.raw, out)
        self.assertEqual({k: sha(p) for k, p in paths.items()}, first)
        self.assertEqual((sha(self.golden), sha(self.raw)), before)
        rows = read_stage(out, "conflicts")
        self.assertEqual(len(rows), 6)
        report = json.loads(paths["field_agreement"].read_text(encoding="utf-8"))
        self.assertEqual(report["records"], 30)
        self.assertEqual(report["conflicts"], 5)  # 5 conflict + 1 pipeline_null = 6 rows
        self.assertEqual(report["fields"]["policy_details.coverage_type"]["agree"], 29)

    def test_missing_input_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            ft_audit.run(self.dir / "nope.jsonl", self.raw, self.dir / "audit")

    def test_record_count_mismatch_is_refused(self):
        self.raw.write_text(json.dumps(RECORDS[:10]), encoding="utf-8")
        with self.assertRaises(ValueError):
            ft_audit.run(self.golden, self.raw, self.dir / "audit")


if __name__ == "__main__":
    unittest.main()
