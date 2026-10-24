import copy
import unittest
from pipeline import run_stages
from pipeline.step11_validate import assemble, validate
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

ADR_GOLDEN = {  # ADR section 8.3, verbatim
    "header": {"claim_id": "CLM-318842", "report_date": "2025-10-06", "incident_date": "2025-10-04",
               "reported_by": "Priya Nair", "channel": "Email"},
    "policy_details": {"policy_number": "POL-301928475", "policyholder_name": "Priya Nair",
                       "coverage_type": "Property", "effective_date": "2024-02-15",
                       "expiration_date": "2026-02-14"},
    "insured_objects": [{"object_id": "PROP-774120", "object_type": "Building", "make_model": "Single Family Home",
                         "year": 1988, "location_address": "58 Lakeshore Rd, Austin", "estimated_value": 410500}],
    "incident_description": {"incident_type": "storm_damage", "location_type": "residence_exterior",
                             "estimated_damage_amount": 11960, "police_report_number": None}}


def stages_for(rec, incident_type, location_type, amount):
    """Run the finished steps in memory; the two LLM steps are stood in by the given labels."""
    st = {}
    prev = lambda s: st[s]
    for name in sorted(run_stages.STAGES):
        st[name] = run_stages.STAGES[name](rec, prev)
    st["08_types"] = {"incident_type": incident_type, "location_type": location_type, "flags": []}
    st["09_damage"] = {"estimated_damage_amount": amount, "flags": []}
    return st


def codes(flags, severity=None):
    return [f["code"] for f in flags if severity is None or f["severity"] == severity]


class Assemble(unittest.TestCase):
    def test_ac1_reference_record_equals_the_adr_golden_output(self):
        st = stages_for({"claim_text": REFERENCE_TEXT}, "storm_damage", "residence_exterior", 11960)
        rec = assemble("rec-001", REFERENCE_TEXT, st)
        self.assertEqual(rec["golden"], ADR_GOLDEN)
        self.assertEqual(list(rec), ["id", "version", "raw_text", "golden", "flags", "review_notes"])
        self.assertEqual((rec["id"], rec["version"], rec["raw_text"], rec["review_notes"]),
                         ("rec-001", "v1.0", REFERENCE_TEXT, ""))
        self.assertEqual(codes(rec["flags"], "Error"), [])

    def test_ac2_all_30_records_have_zero_error_flags_when_the_llm_steps_answer(self):
        for i, r in enumerate(RECORDS):
            d = r["ground_truth"]["incident_description"]
            st = stages_for(r, d["incident_type"], d["location_type"], d["estimated_damage_amount"])
            rec = assemble(f"rec-{i + 1:03d}", r["claim_text"], st)
            self.assertEqual(codes(rec["flags"], "Error"), [], f"record {i}")
            g = rec["golden"]
            self.assertEqual(set(g), {"header", "policy_details", "insured_objects", "incident_description"})
            if g["policy_details"] is None:  # ADR section 2: no policy number -> no policy layers
                self.assertIsNone(g["insured_objects"], f"record {i}")

    def test_known_findings_surface_as_flags_not_silent_fixes(self):
        got = {}
        for i, r in enumerate(RECORDS):
            d = r["ground_truth"]["incident_description"]
            rec = assemble(f"rec-{i + 1:03d}", r["claim_text"],
                           stages_for(r, d["incident_type"], d["location_type"], d["estimated_damage_amount"]))
            for c in codes(rec["flags"]):
                got.setdefault(c, set()).add(i)
        self.assertEqual(got["truncated"], {0, 4, 8, 13})
        self.assertEqual(got["police_date_mismatch"], {0})  # ADR section 9 #1: 2024 incident, PR-2025...
        self.assertEqual(got["holder_inferred"], {9})
        self.assertIn(14, got["not_traceable"])             # the 37098 label (ADR section 9 #2)

    def test_internal_helper_keys_do_not_leak_into_the_golden_record(self):
        r = RECORDS[9]
        st = stages_for(r, "rear_end_collision", "intersection", 2776)
        text = str(assemble("rec-010", r["claim_text"], st)["golden"])
        for leaked in ("holder_inferred", "channel_inferred", "truncated"):
            self.assertNotIn(leaked, text)

    def test_missing_stage_and_unavailable_llm_are_errors_never_silent_nulls(self):
        st = stages_for({"claim_text": REFERENCE_TEXT}, None, None, None)
        del st["08_types"], st["09_damage"]
        rec = assemble("rec-001", REFERENCE_TEXT, st)
        self.assertEqual(codes(rec["flags"], "Error").count("stage_missing"), 2)
        self.assertIn("missing_required", codes(rec["flags"], "Error"))
        st["08_types"] = {"incident_type": None, "location_type": None, "flags": ["llm_unavailable"]}
        st["09_damage"] = {"estimated_damage_amount": None, "flags": ["llm_unavailable"]}
        self.assertEqual(codes(assemble("rec-001", REFERENCE_TEXT, st)["flags"], "Error").count("llm_unavailable"), 2)

    def test_assemble_does_not_modify_its_inputs(self):
        st = stages_for({"claim_text": REFERENCE_TEXT}, "storm_damage", "residence_exterior", 11960)
        before = copy.deepcopy(st)
        assemble("rec-001", REFERENCE_TEXT, st)
        self.assertEqual(st, before)


class Checks(unittest.TestCase):
    """One positive and one negative case per section 6 check, on copies of the ADR golden record."""

    def mutate(self, fn):
        g = copy.deepcopy(ADR_GOLDEN)
        fn(g)
        before = copy.deepcopy(g)
        flags = validate(g)
        self.assertEqual(g, before)  # flag, don't auto-fix
        return flags

    def test_clean_reference_has_no_flags(self):
        self.assertEqual(validate(ADR_GOLDEN, (REFERENCE_TEXT,)), [])

    def test_1_ids_match_regex(self):
        f = self.mutate(lambda g: g["header"].update(claim_id="CLM-12345"))
        self.assertEqual((codes(f), f[0]["severity"]), (["bad_id"], "Error"))
        self.assertEqual(codes(self.mutate(lambda g: g["policy_details"].update(policy_number="POL-1"))), ["bad_id"])
        self.assertEqual(codes(self.mutate(lambda g: g["incident_description"].update(
            police_report_number="PR-2025-1"))), ["bad_id"])
        self.assertEqual(codes(self.mutate(lambda g: g["insured_objects"][0].update(object_id="PROP-x"))), ["bad_id"])
        self.assertEqual(codes(self.mutate(lambda g: g["header"].update(claim_id=None))), [])

    def test_2_enums_in_vocabulary_and_required_ones_not_null(self):
        self.assertEqual(codes(self.mutate(lambda g: g["header"].update(channel="Fax"))), ["bad_enum"])
        self.assertEqual(codes(self.mutate(lambda g: g["incident_description"].update(incident_type="flood"))),
                         ["bad_enum"])
        self.assertEqual(codes(self.mutate(lambda g: g["insured_objects"][0].update(object_type="Boat"))),
                         ["bad_enum", "coverage_object_mismatch"])
        self.assertEqual(codes(self.mutate(lambda g: g["incident_description"].update(location_type=None))),
                         ["missing_required"])

    def test_3_incident_not_after_report(self):
        f = self.mutate(lambda g: g["header"].update(incident_date="2025-10-07"))
        self.assertIn("incident_after_report", codes(f))
        self.assertEqual(codes(self.mutate(lambda g: g["header"].update(incident_date=None))), [])

    def test_4_effective_before_expiration(self):
        f = self.mutate(lambda g: g["policy_details"].update(effective_date="2026-02-14"))
        self.assertEqual(codes(f, "Error"), ["bad_policy_window"])

    def test_5_coverage_matches_object_type(self):
        f = self.mutate(lambda g: g["policy_details"].update(coverage_type="Auto"))
        self.assertEqual(codes(f), ["coverage_object_mismatch"])

    def test_6_amounts_are_positive_integers(self):
        for bad in (0, -1, 12.5, "5"):
            f = self.mutate(lambda g: g["incident_description"].update(estimated_damage_amount=bad))
            self.assertEqual(codes(f), ["bad_amount"], bad)
        f = self.mutate(lambda g: g["insured_objects"][0].update(estimated_value=-3))
        self.assertEqual(codes(f), ["bad_amount"])

    def test_7_police_report_date_close_to_incident_date_is_a_warning(self):
        f = self.mutate(lambda g: g["incident_description"].update(police_report_number="PR-20250830-1234"))
        self.assertEqual((codes(f), f[0]["severity"]), (["police_date_mismatch"], "Warning"))
        f = self.mutate(lambda g: g["incident_description"].update(police_report_number="PR-20251005-1234"))
        self.assertEqual(codes(f), [])  # one day off is fine

    def test_8_incident_outside_policy_window_is_info(self):
        f = self.mutate(lambda g: g["header"].update(incident_date="2023-01-01"))
        self.assertEqual(sorted(codes(f)), ["incident_outside_policy_window"])
        self.assertEqual(f[0]["severity"], "Info")

    def test_9_values_must_be_traceable_to_the_text(self):
        g = copy.deepcopy(ADR_GOLDEN)
        g["header"]["claim_id"] = "CLM-999999"
        g["incident_description"]["estimated_damage_amount"] = 12345
        g["insured_objects"][0]["estimated_value"] = 999
        f = validate(g, (REFERENCE_TEXT,))
        self.assertEqual(sorted(x["field"] for x in f if x["code"] == "not_traceable"),
                         ["header.claim_id", "incident_description.estimated_damage_amount",
                          "insured_objects[0].estimated_value"])
        self.assertTrue(all(x["severity"] == "Warning" for x in f))


if __name__ == "__main__":
    unittest.main()
