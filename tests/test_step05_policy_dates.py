import unittest
from datetime import date
from pipeline.step01_ingest import ingest
from pipeline.step05_policy_dates import add_years, extract_policy_dates as dates
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

# Record 14 says "Valid for 1 year" but its label is +1 year, not the ADR section 4 rule's +1 year - 1 day.
# That is ADR section 9 issue #4, which the owner told us to ignore; we follow the written rule.
KNOWN_MISMATCH = {14}


def pair(eff, exp):
    return {"effective_date": eff, "expiration_date": exp}


class Step05(unittest.TestCase):
    def test_reference_record(self):
        self.assertEqual(dates(REFERENCE_TEXT, "2025-10-06"), pair("2024-02-15", "2026-02-14"))

    def test_all_policy_records_vs_ground_truth(self):
        checked = 0
        for i, r in enumerate(RECORDS):
            pd = r["ground_truth"]["policy_details"]
            if pd is None or i in KNOWN_MISMATCH:
                continue
            got = dates(ingest(r["claim_text"])["text"], r["ground_truth"]["header"]["report_date"])
            self.assertEqual(got, pair(pd["effective_date"], pd["expiration_date"]), f"record {i}")
            checked += 1
        self.assertEqual(checked, 24)

    def test_known_mismatch_is_still_a_mismatch(self):
        r = RECORDS[14]
        got = dates(ingest(r["claim_text"])["text"], r["ground_truth"]["header"]["report_date"])
        self.assertEqual(got, pair("2024-09-25", "2025-09-24"))  # ADR rule: +1y - 1 day
        self.assertNotEqual(got["expiration_date"], r["ground_truth"]["policy_details"]["expiration_date"])

    def test_explicit_full_dates(self):
        self.assertEqual(dates("Effective: 05-Nov-2023  Expires: 04-Nov-2025", None), pair("2023-11-05", "2025-11-04"))
        self.assertEqual(dates("Effective Date: November 19, 2023\nExpiration Date: May 31, 2025", None),
                         pair("2023-11-19", "2025-05-31"))
        self.assertEqual(dates("Policy: POL-1 (liability, Jan 20, 2024 – Jan 19, 2026)", None),
                         pair("2024-01-20", "2026-01-19"))

    def test_partial_dates_anchor_to_report_date(self):
        text = "It started last October 1 and runs until the end of September this year."
        self.assertEqual(dates(text, "2025-09-12"), pair("2024-10-01", "2025-09-30"))
        text = "I've had it since April 18th, I think, 2024, good until October 16th this year."
        self.assertEqual(dates(text, "2025-08-22"), pair("2024-04-18", "2025-10-16"))
        self.assertEqual(dates("went into effect May twenty-seventh this year, expires the 26th two years later",
                               "2025-08-30"), pair("2025-05-27", "2027-05-26"))

    def test_term_rule_and_missing_day(self):
        self.assertEqual(dates("started Aug 1, 2024, good for one year", None), pair("2024-08-01", "2025-07-31"))
        self.assertEqual(dates("since May 2024, good for one year", None), pair("2024-05-01", "2025-04-30"))
        self.assertEqual(add_years(date(2024, 2, 29), 1), date(2025, 2, 28))

    def test_format_hint(self):
        self.assertEqual(dates("POL-776542109 10-06-23/09-06-26 DD/MM/YY (Liability)", None),
                         pair("2023-06-10", "2026-06-09"))
        self.assertEqual(dates("10-06-23/09-06-26 MM/DD/YY", None), pair("2023-10-06", "2026-09-06"))

    def test_null_when_not_derivable(self):
        self.assertEqual(dates("My car was hit yesterday.", "2025-01-01"), pair(None, None))
        self.assertEqual(dates("started in March", None), pair(None, None))  # no year, no anchor
        self.assertEqual(dates("Policy started Jan 5, 2024. The house was valued two years ago.", None),
                         pair("2024-01-05", None))


if __name__ == "__main__":
    unittest.main()
