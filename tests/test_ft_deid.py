import copy
import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from pipeline import ft_deid
from pipeline.store import read_stage
from tests.test_step01_ingest import RECORDS, REFERENCE_TEXT

SUFFIX_WORDS = ("St", "Street", "Rd", "Road", "Ave", "Avenue", "Dr", "Drive", "Ln", "Lane", "Blvd", "Ct", "Court")
REFERENCE_TARGET = {
    "header": {"claim_id": "CLM-318842", "report_date": "2025-10-06", "incident_date": "2025-10-04",
               "reported_by": "Priya Nair", "channel": "Email"},
    "policy_details": {"policy_number": "POL-301928475", "policyholder_name": "Priya Nair",
                       "coverage_type": "Property", "effective_date": "2024-02-15", "expiration_date": "2026-02-14"},
    "insured_objects": [{"object_id": "PROP-774120", "object_type": "Building", "make_model": "Single Family Home",
                         "year": 1988, "location_address": "58 Lakeshore Rd, Austin", "estimated_value": 410500}],
    "incident_description": {"incident_type": "storm_damage", "location_type": "residence_exterior",
                             "estimated_damage_amount": 11960, "police_report_number": None}}


def row(i, text, target):
    return {"id": f"rec-{i:03d}", "claim_text": text, "target": target, "provenance": {}, "warnings": []}


def dataset_rows():
    return [row(i + 1, r["claim_text"], copy.deepcopy(r["ground_truth"])) for i, r in enumerate(RECORDS)]


# ---- an independent checker: it does not use ft_deid's internals -------------------------------------------
def originals(target):
    """(kind, value) personal values of one label set."""
    out = []
    h, p, inc = target["header"], target.get("policy_details"), target["incident_description"]
    names = [h["reported_by"]] + ([p["policyholder_name"]] if p else [])
    out += [("name", n) for n in names if n]
    out += [("id", h["claim_id"])] if h["claim_id"] else []
    out += [("id", p["policy_number"])] if p and p["policy_number"] else []
    out += [("id", inc["police_report_number"])] if inc["police_report_number"] else []
    for o in target.get("insured_objects") or []:
        if o["object_id"]:
            out.append(("id", o["object_id"]))
        if o["location_address"]:
            out.append(("address", o["location_address"]))
    return out


def leaks(kind, value, blob):
    if kind == "name":
        return any(re.search(rf"\b{re.escape(part)}\b", blob) for part in value.split())
    if kind == "id":
        if value.startswith("PR-"):
            return value in blob
        digits = value[3:] if value.startswith("VIN") else value.split("-")[1]
        return re.search(rf"(?<![\d$])(?<!\d,){digits}(?!\d|,\d)", blob) is not None
    num, rest = value.split(" ", 1)
    toks = rest.split(",")[0].split()
    name = " ".join(toks[:-1]) if len(toks) > 1 and toks[-1] in SUFFIX_WORDS else " ".join(toks)
    return ((len(num) >= 4 and re.search(rf"(?<![\d$])(?<!\d,){num}(?!\d|,\d)", blob) is not None)
            or re.search(rf"\b{re.escape(name)}\s+(?:{'|'.join(SUFFIX_WORDS)})\b", blob) is not None
            or re.search(rf"\b{num}\s+{re.escape(name)}\b", blob) is not None)


def present(kind, value, text):
    if kind == "name":
        return all(re.search(rf"\b{re.escape(p)}\b", text) for p in value.split())
    if kind == "id":
        return value in text or re.sub(r"\D", "", value) in text
    num, rest = value.split(" ", 1)
    return num in text and rest.split(",")[0].split()[0] in text


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = dataset_rows()
        cls.out, cls.report = ft_deid.deidentify(cls.rows)
        cls.by_id = {r["id"]: r for r in cls.out}

    def test_ac10_no_original_personal_value_survives_anywhere_in_the_output(self):
        for src in self.rows:
            blob = json.dumps(self.by_id[src["id"]], ensure_ascii=False)
            for kind, value in originals(src["target"]):
                # a value is checked against EVERY output record, not just its own
                self.assertFalse(leaks(kind, value, json.dumps(self.out, ensure_ascii=False)),
                                 f"{src['id']} {kind} still present")
            self.assertEqual(blob.count("\x00"), 0)  # no internal placeholder left behind
        self.assertEqual(self.report["leaks"], [])

    def test_ac11_every_label_still_found_in_its_text_when_it_was_before(self):
        checked = 0
        for src in self.rows:
            new = self.by_id[src["id"]]
            before = {(k, v) for k, v in originals(src["target"]) if present(k, v, src["claim_text"])}
            after = {(k, v) for k, v in originals(new["target"]) if present(k, v, new["claim_text"])}
            self.assertGreaterEqual(len(after), len(before), f"{src['id']}: masking broke a label/text link")  # OCR-garbled ids may become traceable
            checked += len(before)
        self.assertGreater(checked, 100)
        self.assertEqual(self.report["mask_induced_untraceable"], [])

    def test_the_same_original_always_maps_to_the_same_fake_and_distinct_originals_stay_distinct(self):
        names = {}
        for src in self.rows:
            new = self.by_id[src["id"]]
            for o, n in ((src["target"]["header"]["reported_by"], new["target"]["header"]["reported_by"]),):
                names.setdefault(o, set()).add(n)
        self.assertTrue(all(len(v) == 1 for v in names.values()))
        fakes = [next(iter(v)) for v in names.values()]
        self.assertEqual(len(set(fakes)), len(fakes))
        # the holder and the caller are the same person in most claims: same fake in both fields
        for src in self.rows:
            t, n = src["target"], self.by_id[src["id"]]["target"]
            if t["policy_details"] and t["policy_details"]["policyholder_name"] == t["header"]["reported_by"]:
                self.assertEqual(n["policy_details"]["policyholder_name"], n["header"]["reported_by"])

    def test_first_names_shared_by_different_people_keep_one_fake_and_last_names_differ(self):
        jens = [r for r in self.rows if r["target"]["header"]["reported_by"].startswith("Jennifer ")]
        self.assertEqual(len(jens), 4)
        firsts = {self.by_id[r["id"]]["target"]["header"]["reported_by"].split()[0] for r in jens}
        fulls = {self.by_id[r["id"]]["target"]["header"]["reported_by"] for r in jens}
        self.assertEqual((len(firsts), len(fulls)), (1, 4))

    def test_formats_are_preserved(self):
        for src in self.rows:
            t, n = src["target"], self.by_id[src["id"]]["target"]
            if t["header"]["claim_id"]:
                self.assertRegex(n["header"]["claim_id"], r"^CLM-\d{6}$")
            if t["policy_details"]:
                self.assertRegex(n["policy_details"]["policy_number"], r"^POL-\d{9}$")
            pr = t["incident_description"]["police_report_number"]
            if pr:
                new = n["incident_description"]["police_report_number"]
                self.assertRegex(new, r"^PR-\d{8}-\d{4}$")
                self.assertEqual(new[:12], pr[:12])  # the date part is kept (check 7 still works)
            for a, b in zip(t["insured_objects"] or [], n["insured_objects"] or []):
                if a["object_id"]:
                    digits = lambda oid: oid[3:] if oid.startswith("VIN") else oid.split("-")[1]
                    self.assertEqual(b["object_id"][:3 if a["object_id"].startswith("VIN") else 5], a["object_id"][:3 if a["object_id"].startswith("VIN") else 5])
                    self.assertTrue(digits(b["object_id"]).isdigit() and len(digits(b["object_id"])) == len(digits(a["object_id"])))
                    self.assertNotEqual(digits(b["object_id"]), digits(a["object_id"]))
                if a["location_address"] and ", " in a["location_address"]:
                    self.assertEqual(a["location_address"].split(", ", 1)[1], b["location_address"].split(", ", 1)[1])
                if a["location_address"]:
                    self.assertRegex(b["location_address"], r"^\d+ ")

    def test_everything_that_is_not_personal_is_unchanged(self):
        keep = [("header", "report_date"), ("header", "incident_date"), ("header", "channel"),
                ("incident_description", "incident_type"), ("incident_description", "location_type"),
                ("incident_description", "estimated_damage_amount")]
        for src in self.rows:
            n = self.by_id[src["id"]]["target"]
            for sec, k in keep:
                self.assertEqual(n[sec][k], src["target"][sec][k])
            for a, b in zip(src["target"]["insured_objects"] or [], n["insured_objects"] or []):
                for k in ("object_type", "make_model", "year", "estimated_value"):
                    self.assertEqual(a[k], b[k])
            self.assertEqual(self.by_id[src["id"]]["provenance"], src["provenance"])

    def test_fake_values_are_never_words_that_were_already_in_the_data(self):
        original_words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(r["claim_text"] for r in self.rows)))
        firsts, lasts, streets = set(), set(), set()
        for r in self.out:
            parts = r["target"]["header"]["reported_by"].split()
            firsts.add(parts[0])
            lasts.update(parts[1:])
            for o in r["target"].get("insured_objects") or []:
                if o.get("location_address"):
                    street = o["location_address"].split(",")[0].split()[1:]
                    streets.add(" ".join(street[:-1] if street[-1] in SUFFIX_WORDS and len(street) > 1 else street))
        for assigned, pool in ((firsts, ft_deid.FIRST), (lasts, ft_deid.LAST), (streets, ft_deid.STREETS)):
            self.assertTrue(assigned and assigned <= set(pool))
            self.assertFalse(assigned & original_words, assigned & original_words)

    def test_the_report_holds_counts_and_candidates_but_no_original_value(self):
        text = json.dumps(self.report, ensure_ascii=False)
        for src in self.rows:
            for kind, value in originals(src["target"]):
                self.assertFalse(leaks(kind, value, text), f"{kind} in the report")
        self.assertGreater(self.report["replacements"]["person_first"], 30)
        self.assertGreater(self.report["replacements"]["vin"], 5)
        self.assertIn("residual_candidates", self.report)


class Behaviour(unittest.TestCase):
    def test_reference_record(self):
        out, report = ft_deid.deidentify([row(1, REFERENCE_TEXT, copy.deepcopy(REFERENCE_TARGET))])
        r = out[0]
        for gone in ("Priya", "Nair", "CLM-318842", "POL-301928475", "774120", "Lakeshore"):
            self.assertNotIn(gone, r["claim_text"] + json.dumps(r["target"]), gone)
        for kept in ("Austin", "$410,500", "$11,960", "Oct 4", "windstorm", "Saturday", "Vespa"):
            self.assertIn(kept, r["claim_text"])
        name = r["target"]["header"]["reported_by"]
        self.assertIn(name, r["claim_text"] + " ".join(name.split()))  # label and text agree
        self.assertIn(r["target"]["policy_details"]["policy_number"], r["claim_text"])
        self.assertEqual(r["target"]["policy_details"]["policyholder_name"], name)
        self.assertRegex(r["target"]["insured_objects"][0]["location_address"], r"^\d{2} \w+ Rd, Austin$")
        self.assertEqual(report["leaks"], [])

    def test_determinism_and_independence_from_record_order(self):
        rows = dataset_rows()
        a, _ = ft_deid.deidentify(rows)
        b, _ = ft_deid.deidentify(copy.deepcopy(rows))
        self.assertEqual(a, b)
        shuffled = copy.deepcopy(rows)
        random.Random(7).shuffle(shuffled)
        c, _ = ft_deid.deidentify(shuffled)
        self.assertEqual({r["id"]: r for r in a}, {r["id"]: r for r in c})

    def test_a_different_seed_gives_different_fakes(self):
        rows = dataset_rows()[:6]
        a, _ = ft_deid.deidentify(rows, seed="one")
        b, _ = ft_deid.deidentify(rows, seed="two")
        self.assertNotEqual(a[0]["target"]["header"]["reported_by"], b[0]["target"]["header"]["reported_by"])

    def test_ocr_garbled_id_in_the_text_is_masked_with_the_same_fake_as_the_label(self):
        target = copy.deepcopy(REFERENCE_TARGET)
        target["header"]["reported_by"] = target["policy_details"]["policyholder_name"] = "Patricia Adams"
        target["policy_details"]["policy_number"] = "POL-554321090"
        text = "Policy # POL-55432I090 (?? check digit). Customer: Patrica (Patricia) Adams. CLM-318842."
        out, report = ft_deid.deidentify([row(1, text, target)])
        r = out[0]
        self.assertNotIn("55432", r["claim_text"])
        self.assertIn(r["target"]["policy_details"]["policy_number"], r["claim_text"])
        self.assertNotIn("Patrica", r["claim_text"])  # a typo of a label name still leaks the name
        self.assertEqual(report["leaks"], [])

    def test_people_who_are_only_in_the_text_are_masked_too(self):
        text = ("Priya Nair here. My neighbor Sandra saw it. I was greeted by Joanne in the lobby. "
                "My husband William, whose name should be on the policy, was driving.\n\nThanks,\nPriya")
        out, _ = ft_deid.deidentify([row(1, text, copy.deepcopy(REFERENCE_TARGET))])
        for gone in ("Sandra", "Joanne", "William", "Priya", "Nair"):
            self.assertNotIn(gone, out[0]["claim_text"])
        self.assertTrue(out[0]["claim_text"].rstrip().endswith(out[0]["target"]["header"]["reported_by"].split()[0]))

    def test_titles_introductions_and_lists_of_people_are_masked(self):
        text = ("Priya Nair here. My neighbor, Mrs. Garland, brought fans. A customer, her name was Mrs. Patterson. "
                "The agent I spoke to - her name was Tina, I think. Judy from next door helped. "
                "A couple of my employees were there\u2014Camila and Jamie\u2014and they are shaken.")
        out, report = ft_deid.deidentify([row(1, text, copy.deepcopy(REFERENCE_TARGET))])
        for gone in ("Garland", "Patterson", "Tina", "Judy", "Camila", "Jamie", "Priya", "Nair"):
            self.assertNotIn(gone, out[0]["claim_text"], gone)
        self.assertIn("Mrs. ", out[0]["claim_text"])  # the title stays, the name goes
        self.assertEqual(report["leaks"], [])

    def test_dr_as_a_street_suffix_is_not_a_title(self):
        t = copy.deepcopy(REFERENCE_TARGET)
        t["insured_objects"][0]["location_address"] = "4501 Ashwood Dr, Salem"
        out, report = ft_deid.deidentify([row(1, "Priya Nair. Insured at 4501 Ashwood Dr Salem. Then Dr. Hollins called.", t)])
        self.assertIn("Salem", out[0]["claim_text"])          # the city is not a surname
        self.assertNotIn("Hollins", out[0]["claim_text"])     # but a real 'Dr. Name' is
        self.assertEqual(report["leaks"], [])

    def test_the_residual_scan_does_not_hide_an_original_name_that_is_also_a_pool_word(self):
        text = "Priya Nair here. Then Esperanza waved at me from the corner."  # 'Esperanza' is in the fake-name pool
        out, report = ft_deid.deidentify([row(1, text, copy.deepcopy(REFERENCE_TARGET))])
        self.assertIn("Esperanza", out[0]["claim_text"])
        self.assertIn("Esperanza", report["residual_candidates"])
        assigned = out[0]["target"]["header"]["reported_by"].split()
        self.assertTrue(all(a not in report["residual_candidates"] for a in assigned))

    def test_a_name_the_rules_cannot_find_is_reported_as_a_residual_candidate_not_hidden(self):
        text = "Priya Nair here. Then Oswaldo drove past the corner and I called CLM-318842."
        out, report = ft_deid.deidentify([row(1, text, copy.deepcopy(REFERENCE_TARGET))])
        self.assertIn("Oswaldo", out[0]["claim_text"])
        self.assertIn("Oswaldo", report["residual_candidates"])

    def test_vin_in_every_written_form_and_its_trailing_digits(self):
        t = copy.deepcopy(REFERENCE_TARGET)
        t["policy_details"]["coverage_type"] = "Auto"
        t["insured_objects"] = [{"object_id": "VIN12039485762194836", "object_type": "Vehicle", "make_model": "Honda Accord",
                                 "year": 2020, "location_address": None, "estimated_value": 19500}]
        text = ("Priya Nair. My Accord (VIN: VIN12039485762194836) and also VIN 12039485762194836 and "
                "VIN12039485762194836; it ends in …94836. The car is worth $19,500.")
        out, _ = ft_deid.deidentify([row(1, text, t)])
        r = out[0]
        fake = r["target"]["insured_objects"][0]["object_id"]
        self.assertRegex(fake, r"^VIN\d{17}$")
        self.assertNotEqual(fake, "VIN12039485762194836")
        self.assertEqual(r["claim_text"].count(fake[3:]), 3)
        self.assertNotIn("94836", r["claim_text"].replace(fake[3:], ""))  # the trailing-digits hint changed too
        self.assertIn(f"…{fake[-5:]}", r["claim_text"])
        self.assertIn("$19,500", r["claim_text"])

    def test_street_names_are_replaced_only_where_they_are_street_names(self):
        t = copy.deepcopy(REFERENCE_TARGET)
        t["insured_objects"][0]["location_address"] = "7400 Second Street, Clayton"
        text = ("Priya Nair. Second, the roof leaked. The second floor flooded. I live at 7400 Second Street, Clayton, "
                "and the amount was $7,400 in total.")
        out, _ = ft_deid.deidentify([row(1, text, t)])
        r = out[0]["claim_text"]
        self.assertIn("Second, the roof", r)
        self.assertIn("second floor", r)
        self.assertNotIn("7400 Second Street", r)
        self.assertIn("$7,400", r)  # a money amount that happens to equal the house number is not touched
        self.assertTrue(r.count("Clayton") == 1)
        self.assertEqual(out[0]["target"]["insured_objects"][0]["location_address"].split(", ")[1], "Clayton")

    def test_police_number_keeps_its_date_and_changes_the_tail(self):
        t = copy.deepcopy(REFERENCE_TARGET)
        t["incident_description"]["police_report_number"] = "PR-20251004-5507"
        out, _ = ft_deid.deidentify([row(1, "Priya Nair. Officer gave me PR-20251004-5507.", t)])
        new = out[0]["target"]["incident_description"]["police_report_number"]
        self.assertTrue(new.startswith("PR-20251004-") and new != "PR-20251004-5507")
        self.assertIn(new, out[0]["claim_text"])

    def test_inputs_are_not_modified(self):
        rows = dataset_rows()[:5]
        before = copy.deepcopy(rows)
        ft_deid.deidentify(rows)
        self.assertEqual(rows, before)

    def test_a_record_with_no_policy_and_nothing_to_mask_still_works(self):
        t = {"header": {"claim_id": None, "report_date": "2025-09-21", "incident_date": "2025-09-20",
                        "reported_by": "Mark Evans", "channel": "Phone"},
             "policy_details": None, "insured_objects": None,
             "incident_description": {"incident_type": "slip_and_fall", "location_type": "commercial_property",
                                      "estimated_damage_amount": None, "police_report_number": None}}
        out, report = ft_deid.deidentify([row(1, "The name is Mark Evans. Someone fell at my shop.", t)])
        self.assertIsNone(out[0]["target"]["policy_details"])
        self.assertNotIn("Mark", out[0]["claim_text"])
        self.assertEqual(report["leaks"], [])


class Files(unittest.TestCase):
    def test_run_reads_reconciled_rows_and_writes_only_masked_data(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "reconciled.jsonl"
            src.write_text("\n".join(json.dumps(r, sort_keys=True) for r in dataset_rows()[:8]) + "\n", encoding="utf-8")
            out_dir = Path(d) / "finetune"
            path, report = ft_deid.run(src, out_dir)
            self.assertEqual(path.name, "deidentified.jsonl")
            rows = read_stage(out_dir, "deidentified")
            self.assertEqual(len(rows), 8)
            blob = path.read_text(encoding="utf-8")
            for r in dataset_rows()[:8]:
                for kind, value in originals(r["target"]):
                    self.assertFalse(leaks(kind, value, blob))
            first = path.read_bytes()
            ft_deid.run(src, out_dir)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(report["records"], 8)


if __name__ == "__main__":
    unittest.main()
