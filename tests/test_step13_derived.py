import tempfile
import unittest
from pathlib import Path
from pipeline.step13_derived import is_active_at_incident as active, write_derived
from pipeline.store import read_stage
from tests.test_step11_validate import ADR_GOLDEN
from tests.test_step12_output import build


def g(incident, eff, exp, policy=True):
    return {"header": {"incident_date": incident, "claim_id": "CLM-111111"},
            "policy_details": {"effective_date": eff, "expiration_date": exp} if policy else None}


class Step13(unittest.TestCase):
    def test_ac9_inside_outside_and_the_inclusive_boundaries(self):
        self.assertIs(active(g("2025-01-15", "2025-01-01", "2025-12-31")), True)
        self.assertIs(active(g("2025-01-01", "2025-01-01", "2025-12-31")), True)   # first day counts
        self.assertIs(active(g("2025-12-31", "2025-01-01", "2025-12-31")), True)   # last day counts
        self.assertIs(active(g("2024-12-31", "2025-01-01", "2025-12-31")), False)  # day before
        self.assertIs(active(g("2026-01-01", "2025-01-01", "2025-12-31")), False)  # day after

    def test_ac9_null_when_any_of_the_three_dates_is_missing_or_unusable(self):
        for case in (g(None, "2025-01-01", "2025-12-31"), g("2025-06-01", None, "2025-12-31"),
                     g("2025-06-01", "2025-01-01", None), g("2025-06-01", None, None, policy=False),
                     g("not-a-date", "2025-01-01", "2025-12-31"), g("2025-06-01", "2025-13-45", "2025-12-31")):
            self.assertIsNone(active(case), case)

    def test_the_reference_record_is_active(self):
        self.assertIs(active(ADR_GOLDEN), True)  # incident 2025-10-04 inside 2024-02-15..2026-02-14

    def test_golden_schema_is_not_touched(self):
        before = repr(ADR_GOLDEN)
        active(ADR_GOLDEN)
        self.assertEqual(repr(ADR_GOLDEN), before)
        self.assertNotIn("is_active_at_incident", repr(ADR_GOLDEN))

    def test_derived_file_agrees_with_check_8_on_all_30_records(self):
        records = build()
        with tempfile.TemporaryDirectory() as d:
            path = write_derived(records, Path(d) / "final")
            self.assertEqual(path.name, "derived.jsonl")
            rows = read_stage(Path(d) / "final", "derived")
        self.assertEqual(len(rows), 30)
        self.assertEqual(sorted(next(iter(rows.values()))), ["claim_id", "id", "is_active_at_incident"])
        outside = {r["id"] for r in records if any(f["code"] == "incident_outside_policy_window" for f in r["flags"])}
        self.assertEqual({k for k, v in rows.items() if v["is_active_at_incident"] is False}, outside)
        self.assertEqual(len(outside), 5)
        no_policy = {r["id"] for r in records if r["golden"]["policy_details"] is None}
        self.assertEqual(len(no_policy), 5)
        self.assertTrue(no_policy <= {k for k, v in rows.items() if v["is_active_at_incident"] is None})
        self.assertEqual(sum(v["is_active_at_incident"] is True for v in rows.values()), 30 - 5 - len(
            [1 for v in rows.values() if v["is_active_at_incident"] is None]))

    def test_rewrite_is_byte_identical(self):
        records = build(10)
        with tempfile.TemporaryDirectory() as d:
            a = write_derived(records, d).read_bytes()
            self.assertEqual(write_derived(records, d).read_bytes(), a)


if __name__ == "__main__":
    unittest.main()
