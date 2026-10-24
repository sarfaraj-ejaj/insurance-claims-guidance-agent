import unittest
from pipeline.step01_ingest import ingest
from pipeline.step03_claim_id import extract_claim_id
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT


class Step03(unittest.TestCase):
    def test_reference_record(self):
        self.assertEqual(extract_claim_id(REFERENCE_TEXT), "CLM-318842")

    def test_all_30_records_match_ground_truth(self):
        for i, r in enumerate(RECORDS):
            got = extract_claim_id(ingest(r["claim_text"])["text"])
            self.assertEqual(got, r["ground_truth"]["header"]["claim_id"], f"record {i}")

    def test_missing_gives_null(self):
        self.assertIsNone(extract_claim_id("no id here, maybe POL- or PLO-"))

    def test_malformed_ids_are_rejected(self):
        self.assertIsNone(extract_claim_id("my claim ref is CLM-12345"))      # too short
        self.assertIsNone(extract_claim_id("my claim ref is CLM-1234567"))    # too long
        self.assertIsNone(extract_claim_id("claim CLM-12345x"))

    def test_prefers_id_near_claim_label(self):
        text = "Earlier claim CLM-111111 was closed. My claim number is CLM-222222 today."
        self.assertEqual(extract_claim_id(text), "CLM-222222")
        text = "Claim reference: CLM-333333. My neighbour had CLM-444444."
        self.assertEqual(extract_claim_id(text), "CLM-333333")

    def test_falls_back_to_first_and_uppercases(self):
        self.assertEqual(extract_claim_id("see CLM-555555 and CLM-666666"), "CLM-555555")
        self.assertEqual(extract_claim_id("claim ref clm-777777"), "CLM-777777")


if __name__ == "__main__":
    unittest.main()
