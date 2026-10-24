import unittest
from pipeline.step01_ingest import ingest
from pipeline.step07_incident_date import extract_incident_date as inc
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

REPORT = "2025-09-04"  # a Thursday


class Step07(unittest.TestCase):
    def test_reference_record(self):
        self.assertEqual(inc(REFERENCE_TEXT, "2025-10-06"), ("2025-10-04", []))

    def test_all_30_records_match_ground_truth(self):
        for i, r in enumerate(RECORDS):
            h = r["ground_truth"]["header"]
            got, _ = inc(ingest(r["claim_text"])["text"], h["report_date"])
            self.assertEqual(got, h["incident_date"], f"record {i}")

    def test_relative_phrases(self):
        for text, want in (("The accident happened yesterday.", "2025-09-03"),
                           ("The fire happened this morning.", "2025-09-04"),
                           ("It happened two days ago.", "2025-09-02"),
                           ("The crash was 3 days ago.", "2025-09-01"),
                           ("The fire happened the day before I reported it.", "2025-09-03")):
            self.assertEqual(inc(text, REPORT), (want, []), text)

    def test_today_means_report_date(self):
        self.assertEqual(inc("The incident actually happened earlier today.", REPORT)[0], REPORT)

    def test_explicit_date_beats_relative_and_flags_conflict(self):
        got = inc("The storm happened September 2nd, though last night was also windy and the accident hit yesterday.",
                  REPORT)
        self.assertEqual(got, ("2025-09-02", ["incident_date_conflict"]))

    def test_weekday_with_day_of_month_is_verified(self):
        self.assertEqual(inc("The accident was Tuesday the 2nd.", REPORT), ("2025-09-02", []))
        # Wednesday the 3rd is right; 'Thursday the 3rd' is not a Thursday -> the number wins, flagged
        self.assertEqual(inc("The accident was Thursday the 3rd.", REPORT), ("2025-09-03", ["weekday_mismatch"]))
        self.assertEqual(inc("The accident was Monday the 1st.", "2025-09-04"), ("2025-09-01", []))

    def test_bare_weekday_is_most_recent_before_report(self):
        self.assertEqual(inc("The accident happened Sunday evening.", REPORT)[0], "2025-08-31")
        self.assertEqual(inc("The accident happened Thursday.", REPORT)[0], "2025-08-28")  # not the report day

    def test_day_first_and_labelled_dates(self):
        self.assertEqual(inc("INCIDENT: ~WATER DMG, burst pipe, 17th Sep.", "2025-09-19")[0], "2025-09-17")
        self.assertEqual(inc("Incident Details:\nDate: August 30, 2025\nLocation: lot", "2025-08-30")[0], "2025-08-30")

    def test_year_is_inferred_from_report_date(self):
        self.assertEqual(inc("The storm happened Dec 30.", "2026-01-02")[0], "2025-12-30")

    def test_non_incident_sentences_are_ignored(self):
        self.assertEqual(inc("I called in yesterday. A contractor came yesterday. Mechanic is free Thursday.", REPORT),
                         (None, []))
        self.assertEqual(inc("The accident happened yesterday. Today's the 4th.", REPORT)[0], "2025-09-03")

    def test_policy_dates_and_old_dates_are_not_incident_dates(self):
        text = "The accident happened yesterday, and the policy expires May 13th."
        self.assertEqual(inc(text, REPORT)[0], "2025-09-03")
        text = "I had this thing happen at my house, built on January 1st, 1959."
        self.assertEqual(inc(text, REPORT), (None, []))  # a 1959 date is not a plausible incident
        self.assertEqual(inc("The accident happened yesterday.", None), (None, []))
        self.assertEqual(inc("The accident happened on September 20th.", REPORT), (None, []))  # in the future

    def test_system_tags_do_not_count(self):
        self.assertEqual(inc("The incident happened [System timestamp: 2025-09-01] yesterday.", REPORT)[0], "2025-09-03")


if __name__ == "__main__":
    unittest.main()
