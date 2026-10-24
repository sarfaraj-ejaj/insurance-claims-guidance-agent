import unittest
from pipeline.step01_ingest import ingest
from pipeline.step06_objects import extract_objects
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

# (record, field) pairs where the label is not derivable from the text; accepted and documented.
#  year/4: "built the same year as my birthday"; value/7 and value/12: label differs from the stated
#  figure ("$372,000" vs 372760, "a bit under $400,000" vs 392607) - ADR section 9 issues, ignored by the owner.
KNOWN_GAPS = {("year", 4), ("estimated_value", 7), ("estimated_value", 12)}
REFERENCE_OBJECT = {"object_id": "PROP-774120", "object_type": "Building", "make_model": "Single Family Home",
                    "year": 1988, "location_address": "58 Lakeshore Rd, Austin", "estimated_value": 410500}


def key(o):
    return o["object_id"] or ""


class Step06(unittest.TestCase):
    def test_reference_record_matches_adr_golden(self):
        self.assertEqual(extract_objects(REFERENCE_TEXT, "Property"), [REFERENCE_OBJECT])

    def test_policy_records_vs_ground_truth(self):
        seen_gaps, checked = set(), 0
        for i, r in enumerate(RECORDS):
            pd = r["ground_truth"]["policy_details"]
            if pd is None:
                continue
            want = sorted(r["ground_truth"]["insured_objects"], key=key)
            got = sorted(extract_objects(ingest(r["claim_text"])["text"], pd["coverage_type"]), key=key)
            self.assertEqual(len(got), len(want), f"record {i}: object count")
            for g, w in zip(got, want):
                for k, v in w.items():
                    if (k, i) in KNOWN_GAPS:
                        seen_gaps.add((k, i))
                        self.assertNotEqual(g[k], v, f"record {i} {k}: gap closed, remove it from KNOWN_GAPS")
                    else:
                        self.assertEqual(g[k], v, f"record {i} {k}")
            checked += 1
        self.assertEqual(seen_gaps, KNOWN_GAPS)
        self.assertEqual(checked, 25)

    def test_no_coverage_means_no_objects(self):
        self.assertIsNone(extract_objects("anything", None))

    def test_damage_quotes_are_not_values(self):
        o = extract_objects("around $5,800 worth of repairs. Estimate of damages is over $37,000.", "Property")[0]
        self.assertIsNone(o["estimated_value"])
        o = extract_objects("The shop said the damage came to half the car's value. Their quote was $7,800. "
                            "The Mazda's value is around $15.6k. VIN77889900112233445", "Auto")[0]
        self.assertEqual(o["estimated_value"], 15600)

    def test_each_vehicle_gets_its_own_value_and_duplicates_merge(self):
        text = ("I was driving my 2019 Mercedes GLC (VIN: VIN19157663903019200) when hit.\n"
                "1. 2017 BMW X3 (VIN: VIN95384130612982136) - estimated value $15,265\n"
                "2. 2019 Mercedes GLC (VIN: VIN19157663903019200) - estimated value $20,372")
        objs = sorted(extract_objects(text, "Auto"), key=key)
        self.assertEqual([(o["object_id"], o["make_model"], o["year"], o["estimated_value"]) for o in objs],
                         [("VIN19157663903019200", "Mercedes GLC", 2019, 20372),
                          ("VIN95384130612982136", "BMW X3", 2017, 15265)])

    def test_vehicle_without_vin_and_uncertain_make(self):
        o = extract_objects("My car is an old beater from \u201999, maybe a Ford or Mazda? [2025-09-22]", "Auto")[0]
        self.assertEqual((o["object_id"], o["make_model"], o["year"]), (None, None, 1999))

    def test_building_id_prefix_follows_coverage(self):
        self.assertEqual(extract_objects("liability is under id 982311 in the system", "Liability")[0]["object_id"],
                         "LIAB-982311")
        self.assertEqual(extract_objects("number 635143 in your system", "Property")[0]["object_id"], "PROP-635143")
        self.assertEqual(extract_objects("system id PROP-819304", "Property")[0]["object_id"], "PROP-819304")
        self.assertIsNone(extract_objects("claim CLM-424063 filed", "Property")[0]["object_id"])

    def test_address_from_fragments_and_building_year(self):
        o = extract_objects("I live over on Church Street in Georgetown, house number 9600, built \u201996.", "Property")[0]
        self.assertEqual((o["location_address"], o["year"]), ("9600 Church Street, Georgetown", 1996))
        o = extract_objects("my townhouse at 441 Oakwood Avenue in Buffalo. The building is from 2008.", "Property")[0]
        self.assertEqual((o["location_address"], o["year"], o["make_model"]), ("441 Oakwood Avenue, Buffalo", 2008, "Townhouse"))


if __name__ == "__main__":
    unittest.main()
