import unittest
from pipeline.step01_ingest import ingest
from pipeline.step10_police import extract_police_report_number as pr
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT


class Step10(unittest.TestCase):
    def test_reference_record_has_no_police_report(self):
        self.assertIsNone(pr(REFERENCE_TEXT))  # "No police, but the city arborist came by."

    def test_all_30_records_match_ground_truth(self):
        found = 0
        for i, r in enumerate(RECORDS):
            want = r["ground_truth"]["incident_description"]["police_report_number"]
            self.assertEqual(pr(ingest(r["claim_text"])["text"]), want, f"record {i}")
            found += want is not None
        self.assertEqual(found, 14)

    def test_valid_number(self):
        self.assertEqual(pr("the officer gave me PR-20250913-5507, I think."), "PR-20250913-5507")
        self.assertEqual(pr("report (pr-20250913-5507)"), "PR-20250913-5507")  # IDs are uppercase

    def test_none_or_garbled_is_null(self):
        for text in ("No police number\u2014I wasn't sure.", "the number was PR-2025091-5507", "PR-20250913-55071",
                     "PR-20250913-550", "PR-2025O913-5507", "PR-20250913", "system id PROP-819304", ""):
            self.assertIsNone(pr(text), text)

    def test_prefers_number_near_police_wording(self):
        text = "A reference PR-20250101-1111 was on the form. The officer gave me PR-20250913-5507."
        self.assertEqual(pr(text), "PR-20250913-5507")
        self.assertEqual(pr("see PR-20250101-1111 and PR-20250202-2222"), "PR-20250101-1111")  # no cue: first


if __name__ == "__main__":
    unittest.main()
