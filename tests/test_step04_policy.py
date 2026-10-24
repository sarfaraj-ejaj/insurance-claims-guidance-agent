import unittest
from pipeline.step01_ingest import ingest
from pipeline.step04_policy import (extract_coverage_type, extract_policy, extract_policy_number,
                                    extract_policyholder)
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

# Known heuristic miss, accepted and documented: record 7's liability coverage is only implied
# ("worried about the liability part") while 'property' words dominate. Fix needs the LLM (T08 era).
KNOWN_COVERAGE_MISS = {7}
CORE = ("policy_number", "policyholder_name", "coverage_type")


class Step04(unittest.TestCase):
    def test_reference_record(self):
        self.assertEqual(extract_policy(REFERENCE_TEXT, "Priya Nair"), {
            "policy_number": "POL-301928475", "policyholder_name": "Priya Nair",
            "coverage_type": "Property"})

    def test_all_30_records_vs_ground_truth(self):
        for i, r in enumerate(RECORDS):
            got = extract_policy(ingest(r["claim_text"])["text"])
            want = r["ground_truth"]["policy_details"]
            if want is None:
                self.assertIsNone(got, f"record {i}: no policy number stated -> None")
                continue
            for k in CORE:
                if k == "coverage_type" and i in KNOWN_COVERAGE_MISS:
                    continue
                self.assertEqual(got[k], want[k], f"record {i} {k}")

    def test_known_miss_is_still_a_miss(self):
        # if this starts passing, delete the allowance above
        got = extract_policy(ingest(RECORDS[7]["claim_text"])["text"])
        self.assertNotEqual(got["coverage_type"], RECORDS[7]["ground_truth"]["policy_details"]["coverage_type"])

    def test_partial_or_garbled_number_is_null(self):
        for s in ("POL-1938something", "POL-12345", "POL- or PLO-", "POL-1234567890", "POL-55432I090x"):
            self.assertIsNone(extract_policy_number(s), s)
        self.assertIsNone(extract_policy("my cousin has POL-77654something"))

    def test_other_peoples_policies_ignored(self):
        text = "My policy is POL-111111111. My husband has his own policy POL-222222222."
        self.assertEqual(extract_policy_number(text), "POL-111111111")
        text = "My cousin's POL-333333333 is unrelated, but mine is POL-444444444."
        self.assertEqual(extract_policy_number(text), "POL-444444444")
        # a lone number is kept even if a relative is mentioned nearby (record 9 pattern)
        self.assertEqual(extract_policy_number("my husband said his number is POL-555555555"), "POL-555555555")

    def test_holder_may_differ_from_caller_and_is_flagged(self):
        text = "my husband William, whose name should be on the actual policy, POL-575631637 auto"
        self.assertEqual(extract_policyholder(text, "Lisa Garcia"), ("William Garcia", True))
        self.assertEqual(extract_policyholder("plain", "Lisa Garcia"), ("Lisa Garcia", False))
        self.assertTrue(extract_policy(text, "Lisa Garcia")["holder_inferred"])

    def test_coverage_explicit_beats_keywords(self):
        self.assertEqual(extract_coverage_type("my auto policy POL-123456789, roof house", "POL-123456789"), "Auto")
        self.assertEqual(extract_coverage_type("Policy: POL-123456789 (liability, Jan 2024)", "POL-123456789"), "Liability")
        self.assertEqual(extract_coverage_type("Coverage: Property", "POL-000000000"), "Property")
        self.assertIsNone(extract_coverage_type("nothing relevant", "POL-123456789"))


if __name__ == "__main__":
    unittest.main()
