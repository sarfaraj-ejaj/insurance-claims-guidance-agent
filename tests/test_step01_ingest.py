"""Also hosts the shared fixtures (ADR section 8 reference sample + the 30 records) that later tests import."""
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = json.loads((ROOT / "insurance_claims.json").read_text(encoding="utf-8"))

REFERENCE_TEXT = """[Email received: 2025-10-06]
Hi, Priya Nair here. My claim ref is CLM-318842. During Saturday's windstorm (Oct 4, around 7pm)
a big oak came down on the garage of my house at 58 Lakeshore Rd, Austin. It was built in '88 and
the county valued it around $410,500 (our system id is 774120). My property policy is POL-301928475,
started Feb 15, 2024, and runs two years. The contractor said "about 12k" but the written quote is
$11,960. No police, but the city arborist came by. My old Vespa is insured elsewhere, ignore that."""

from pipeline.step01_ingest import ingest

# records whose text ends mid-sentence (checked by reading each ending)
TRUNCATED = {0, 4, 8, 13}


class Step01(unittest.TestCase):
    def test_raw_text_untouched_and_tags_kept(self):
        for r in RECORDS:
            out = ingest(r["claim_text"])
            self.assertEqual(out["raw_text"], r["claim_text"])
        out = ingest(REFERENCE_TEXT)
        self.assertEqual(out["text"], REFERENCE_TEXT)
        self.assertTrue(out["text"].startswith("[Email received: 2025-10-06]"))
        self.assertFalse(out["truncated"])

    def test_truncated_flag_on_dataset(self):
        got = {i for i, r in enumerate(RECORDS) if ingest(r["claim_text"])["truncated"]}
        self.assertEqual(got, TRUNCATED)

    def test_truncated_synthetic(self):
        self.assertTrue(ingest("I was driving home when")["truncated"])
        self.assertTrue(ingest("It was at house number, ohh right, it's 7400 Pi..")["truncated"])
        self.assertFalse(ingest("All done.\n\n---")["truncated"])
        self.assertFalse(ingest("Is it covered?")["truncated"])
        self.assertFalse(ingest("Thanks,\nMichelle")["truncated"])
        self.assertFalse(ingest("Hello.\n\n[Email received: 2025-08-26]")["truncated"])

    def test_ocr_fixed_only_in_ids_when_unambiguous(self):
        self.assertIn("POL-554321090", ingest("Policy # POL-55432I090 (?? check)")["text"])
        self.assertIn("CLM-773011", ingest("REF: CLM-77301l")["text"])
        self.assertIn("CLM-100200", ingest("REF: CLM-1OO2OO")["text"])
        # not an OCR slip: leave alone
        self.assertIn("POL-77654something", ingest("POL-77654something")["text"])
        self.assertIn("POL-1938something", ingest("POL-1938something")["text"])
        # prose with I / l is never touched
        self.assertEqual(ingest("I live in Illinois.")["text"], "I live in Illinois.")

    def test_obvious_name_typo_with_correction(self):
        self.assertIn("Patricia Adams", ingest("Customer: Patrica (Patricia) Adams")["text"])
        self.assertEqual(ingest("Call me (Bob)")["text"], "Call me (Bob)")

    def test_reference_ingest_flags(self):
        out = ingest(REFERENCE_TEXT)
        self.assertEqual(set(out), {"raw_text", "text", "truncated"})


if __name__ == "__main__":
    unittest.main()
