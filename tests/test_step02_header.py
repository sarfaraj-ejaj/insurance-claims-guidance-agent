import unittest
from pipeline.step01_ingest import ingest
from pipeline.step02_header import (extract_channel, extract_header, extract_report_date,
                                    extract_reported_by, parse_date)
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT


class Step02(unittest.TestCase):
    def test_reference_record(self):
        self.assertEqual(extract_header(REFERENCE_TEXT), {
            "channel": "Email", "report_date": "2025-10-06", "reported_by": "Priya Nair"})

    def test_all_30_records_match_ground_truth(self):
        for i, r in enumerate(RECORDS):
            got = extract_header(ingest(r["claim_text"])["text"])
            want = r["ground_truth"]["header"]
            for k in ("channel", "report_date", "reported_by"):
                self.assertEqual(got[k], want[k], f"record {i} {k}")

    def test_date_formats(self):
        for s, iso in (("2025-09-04", "2025-09-04"), ("16-8-2025", "2025-08-16"),
                       ("19-SEP-2025", "2025-09-19"), ("Sun, Sep 21, 2025"[5:], "2025-09-21"),
                       ("August 22nd 2025", "2025-08-22")):
            self.assertEqual(parse_date(s), iso)
        with self.assertRaises(ValueError):
            parse_date("31-02-2025")
        with self.assertRaises(ValueError):
            parse_date("next Tuesday")

    def test_report_date_sources(self):
        self.assertEqual(extract_report_date("Submission timestamp: 16-8-2025. Submission id"), "2025-08-16")
        self.assertEqual(extract_report_date("Email received: Sun, Sep 21, 2025\n\nFrom"), "2025-09-21")
        self.assertEqual(extract_report_date("x" * 300 + " [System timestamp: 2025-08-20] y"), "2025-08-20")
        self.assertIsNone(extract_report_date("no dates here"))

    def test_channel_tag_vs_inferred(self):
        self.assertEqual(extract_channel("[Portal submission: 2025-09-15]"), ("Portal", False))
        self.assertEqual(extract_channel("Hi, I'm calling about a claim"), ("Phone", True))
        self.assertEqual(extract_channel("I'm emailing you today"), ("Email", True))
        self.assertEqual(extract_channel("I dropped by your office"), ("In-Person", True))
        self.assertEqual(extract_channel("nothing useful"), (None, True))
        self.assertTrue(extract_header("I'm calling you")["channel_inferred"])
        self.assertNotIn("channel_inferred", extract_header("[Email received: 2025-10-06]"))

    def test_reporter_not_confused_with_dates(self):
        self.assertIsNone(extract_reported_by("This is Saturday September 20th"))
        self.assertEqual(extract_reported_by("Caller: Mark Evans"), "Mark Evans")
        self.assertEqual(extract_reported_by("Submitted by Jennifer Johnson"), "Jennifer Johnson")


if __name__ == "__main__":
    unittest.main()
