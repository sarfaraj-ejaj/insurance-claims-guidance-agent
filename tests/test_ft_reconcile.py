import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from pipeline import ft_reconcile as rc
from pipeline.store import read_stage
from tests.test_ft_audit import golden_rows
from tests.test_step01_ingest import RECORDS

CONFLICTED = ["rec-005", "rec-008", "rec-013", "rec-015"]


def d(record, field, choose, **kw):
    return {"record": record, "field": field, "choose": choose, "by": "reviewer", "reason": "checked the text", **kw}


ALL_FOR_008 = [d("rec-008", "policy_details.coverage_type", "dataset"),
               d("rec-008", "insured_objects[0].make_model", "dataset"),
               d("rec-008", "insured_objects[0].estimated_value", "value", value=372000)]


def by_id(rows):
    return {r["id"]: r for r in rows}


class Reconcile(unittest.TestCase):
    def test_conflicting_records_are_excluded_with_their_reasons(self):
        ready, excluded, _ = rc.reconcile(golden_rows(), RECORDS, [])
        self.assertEqual(sorted(r["id"] for r in excluded), CONFLICTED)
        self.assertEqual(len(ready), 26)
        reasons = {e["id"]: sorted(x["field"] for x in e["reasons"]) for e in excluded}
        self.assertEqual(reasons["rec-008"], ["insured_objects[0].estimated_value", "insured_objects[0].make_model",
                                              "policy_details.coverage_type"])
        self.assertEqual(reasons["rec-015"], ["policy_details.expiration_date"])
        x = next(x for x in by_id(excluded)["rec-015"]["reasons"])
        self.assertEqual((x["kind"], x["dataset"]), ("conflict", RECORDS[14]["ground_truth"]["policy_details"]["expiration_date"]))

    def test_a_pipeline_null_is_not_a_conflict_the_dataset_label_is_used(self):
        rows = golden_rows()
        for r in rows:  # the real situation today: the Claude steps have not run
            r["golden"]["incident_description"].update(incident_type=None, location_type=None)
            r["flags"].append({"code": "llm_unavailable", "severity": "Error", "field": "08_types", "message": ""})
        ready, excluded, _ = rc.reconcile(rows, RECORDS, [])
        self.assertEqual(len(ready), 26)
        r = by_id(ready)["rec-001"]
        self.assertEqual(r["provenance"]["incident_description.incident_type"], "dataset_only")
        self.assertEqual(r["target"]["incident_description"]["incident_type"],
                         RECORDS[0]["ground_truth"]["incident_description"]["incident_type"])
        self.assertEqual(r["provenance"]["header.claim_id"], "agree")

    def test_targets_are_the_dataset_labels_and_the_inputs_are_untouched(self):
        rows, raw = golden_rows(), copy.deepcopy(RECORDS)
        before = (copy.deepcopy(rows), copy.deepcopy(raw))
        ready, _, _ = rc.reconcile(rows, raw, [])
        self.assertEqual((rows, raw), before)
        for r in ready:
            want = copy.deepcopy(raw[int(r["id"][4:]) - 1]["ground_truth"])
            if want["insured_objects"]:
                want["insured_objects"] = rc._objects(want)
            self.assertEqual(r["target"], want)
            self.assertEqual(r["claim_text"], raw[int(r["id"][4:]) - 1]["claim_text"])

    def test_decisions_bring_records_back_with_the_chosen_value(self):
        ready, excluded, ignored = rc.reconcile(golden_rows(), RECORDS, [d("rec-015", "policy_details.expiration_date", "pipeline")])
        self.assertEqual(ignored, [])
        self.assertEqual(len(ready), 27)
        r = by_id(ready)["rec-015"]
        self.assertEqual(r["provenance"]["policy_details.expiration_date"], "decision")
        self.assertEqual(r["target"]["policy_details"]["expiration_date"], "2025-09-24")  # the pipeline's value
        ready, excluded, _ = rc.reconcile(golden_rows(), RECORDS, ALL_FOR_008)
        r = by_id(ready)["rec-008"]
        self.assertEqual(r["target"]["policy_details"]["coverage_type"], "Liability")
        self.assertEqual(r["target"]["insured_objects"][0]["estimated_value"], 372000)  # a typed value
        self.assertEqual(r["provenance"]["insured_objects[0].estimated_value"], "decision")

    def test_a_partial_decision_leaves_the_record_excluded(self):
        _, excluded, _ = rc.reconcile(golden_rows(), RECORDS, ALL_FOR_008[:2])
        e = by_id(excluded)["rec-008"]
        self.assertEqual([x["field"] for x in e["reasons"]], ["insured_objects[0].estimated_value"])

    def test_pipeline_only_value_keeps_the_dataset_null_and_warns(self):
        rows = golden_rows()
        rows[0]["golden"]["header"]["claim_id"] = "CLM-424063"
        raw = copy.deepcopy(RECORDS)
        raw[0]["ground_truth"]["header"]["claim_id"] = None
        ready, _, _ = rc.reconcile(rows, raw, [])
        r = by_id(ready)["rec-001"]
        self.assertIsNone(r["target"]["header"]["claim_id"])
        self.assertEqual(r["provenance"]["header.claim_id"], "pipeline_only")
        self.assertEqual(r["warnings"], ["pipeline_only:header.claim_id"])

    def test_object_count_conflict_can_be_decided_for_the_pipeline_list(self):
        rows = golden_rows()
        rows[3]["golden"]["insured_objects"] = rows[3]["golden"]["insured_objects"][:1]
        _, excluded, _ = rc.reconcile(rows, RECORDS, [])
        self.assertIn("rec-004", [e["id"] for e in excluded])
        ready, _, _ = rc.reconcile(rows, RECORDS, [d("rec-004", "insured_objects", "pipeline")])
        self.assertEqual(len(by_id(ready)["rec-004"]["target"]["insured_objects"]), 1)

    def test_bad_decisions_are_refused_and_stale_ones_are_reported(self):
        bad = [d("rec-015", "policy_details.expiration_date", "maybe"),
               {"record": "rec-015", "field": "policy_details.expiration_date", "choose": "value", "by": "x", "reason": "y"},
               d("rec-999", "header.claim_id", "dataset"), d("rec-001", "header.nope", "dataset"),
               {"record": "rec-015", "field": "policy_details.expiration_date", "choose": "dataset"}]  # no by/reason
        for decision in bad:
            with self.assertRaises(ValueError, msg=str(decision)):
                rc.reconcile(golden_rows(), RECORDS, [decision])
        _, _, ignored = rc.reconcile(golden_rows(), RECORDS, [d("rec-001", "header.claim_id", "dataset")])
        self.assertEqual(ignored, [{"record": "rec-001", "field": "header.claim_id", "why": "no conflict on this field"}])

    def test_a_null_from_a_claude_step_that_did_run_is_a_conflict_not_dataset_only(self):
        rows = golden_rows()
        rows[14]["golden"]["incident_description"]["estimated_damage_amount"] = None  # Claude ran, found no support
        ready, excluded, _ = rc.reconcile(rows, RECORDS, [d("rec-015", "policy_details.expiration_date", "pipeline")])
        self.assertIn("rec-015", [e["id"] for e in excluded])
        fields = [x["field"] for x in by_id(excluded)["rec-015"]["reasons"]]
        self.assertEqual(fields, ["incident_description.estimated_damage_amount"])
        ready, _, _ = rc.reconcile(rows, RECORDS, [d("rec-015", "policy_details.expiration_date", "pipeline"),
                                                   d("rec-015", "incident_description.estimated_damage_amount", "value", value=37000)])
        self.assertEqual(by_id(ready)["rec-015"]["target"]["incident_description"]["estimated_damage_amount"], 37000)


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.golden = self.dir / "golden.jsonl"
        self.golden.write_text("\n".join(json.dumps(r, sort_keys=True) for r in golden_rows()) + "\n", encoding="utf-8")
        self.raw = self.dir / "raw.json"
        self.raw.write_text(json.dumps(RECORDS), encoding="utf-8")
        self.decisions = self.dir / "review" / "decisions.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def sha(self, p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def test_load_decisions(self):
        self.assertEqual(rc.load_decisions(self.dir / "absent.jsonl"), [])  # no file = no decisions
        self.decisions.parent.mkdir()
        self.decisions.write_text(json.dumps(ALL_FOR_008[0]) + "\n\n" + json.dumps(ALL_FOR_008[1]) + "\n", encoding="utf-8")
        self.assertEqual(len(rc.load_decisions(self.decisions)), 2)
        self.decisions.write_text('{"ok": 1}\n{broken\n', encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            rc.load_decisions(self.decisions)
        self.assertIn("line 2", str(cm.exception))

    def test_run_writes_ready_and_excluded_files_deterministically(self):
        out = self.dir / "finetune"
        paths = rc.run(self.golden, self.raw, self.decisions, out)
        ready, excluded = read_stage(out, "reconciled"), read_stage(out, "excluded")
        self.assertEqual((len(ready), sorted(excluded)), (26, CONFLICTED))
        self.assertEqual(sorted(next(iter(ready.values()))), ["claim_text", "id", "provenance", "target", "warnings"])
        first = {k: self.sha(p) for k, p in paths.items()}
        rc.run(self.golden, self.raw, self.decisions, out)
        self.assertEqual({k: self.sha(p) for k, p in paths.items()}, first)

    def test_run_reads_the_decisions_file_and_never_writes_it(self):
        self.decisions.parent.mkdir()
        self.decisions.write_text("\n".join(json.dumps(x) for x in ALL_FOR_008) + "\n", encoding="utf-8")
        before = self.sha(self.decisions)
        rc.run(self.golden, self.raw, self.decisions, self.dir / "finetune")
        self.assertEqual(len(read_stage(self.dir / "finetune", "reconciled")), 27)
        self.assertEqual(self.sha(self.decisions), before)


if __name__ == "__main__":
    unittest.main()
