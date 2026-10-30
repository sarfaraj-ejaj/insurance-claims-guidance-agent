import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from pipeline import ft_deid, ft_export as fe
from pipeline.step00_schema import SCHEMA, VOCAB
from tests.test_ft_deid import REFERENCE_TARGET, dataset_rows, row
from tests.test_step01_ingest import REFERENCE_TEXT

KEYS = {sec: list(fields) for sec, fields in SCHEMA.items()}


def good():
    return copy.deepcopy(REFERENCE_TARGET)


def problems(mutator):
    t = good()
    mutator(t)
    return fe.validate_target(t)


class Validate(unittest.TestCase):
    def test_a_valid_target_has_no_problems(self):
        self.assertEqual(fe.validate_target(good()), [])

    def test_each_kind_of_defect_is_caught(self):
        cases = {
            "bad enum": lambda t: t["incident_description"].update(incident_type="flood"),
            "null required enum": lambda t: t["incident_description"].update(location_type=None),
            "bad channel": lambda t: t["header"].update(channel="Fax"),
            "bad claim id": lambda t: t["header"].update(claim_id="CLM-12345"),
            "bad policy id": lambda t: t["policy_details"].update(policy_number="POL-1"),
            "bad police id": lambda t: t["incident_description"].update(police_report_number="PR-1-2"),
            "bad object id": lambda t: t["insured_objects"][0].update(object_id="HOUSE-1"),
            "bad date": lambda t: t["header"].update(report_date="10/06/2025"),
            "float amount": lambda t: t["incident_description"].update(estimated_damage_amount=11960.5),
            "negative amount": lambda t: t["insured_objects"][0].update(estimated_value=-1),
            "bool as number": lambda t: t["insured_objects"][0].update(year=True),
            "string number": lambda t: t["incident_description"].update(estimated_damage_amount="11960"),
            "extra key": lambda t: t["header"].update(review_notes="x"),
            "missing key": lambda t: t["header"].pop("channel"),
            "extra section": lambda t: t.update(flags=[]),
            "policy null but objects": lambda t: t.update(policy_details=None),
            "objects null but policy": lambda t: t.update(insured_objects=None),
            "empty string": lambda t: t["header"].update(reported_by=" "),
            "bad object type": lambda t: t["insured_objects"][0].update(object_type="Boat"),
        }
        for name, mutate in cases.items():
            self.assertTrue(problems(mutate), name)

    def test_a_claim_with_no_policy_is_valid_when_both_layers_are_null(self):
        t = good()
        t["policy_details"] = t["insured_objects"] = None
        self.assertEqual(fe.validate_target(t), [])

    def test_nullable_fields_may_be_null(self):
        t = good()
        t["header"].update(claim_id=None, incident_date=None)
        t["policy_details"].update(effective_date=None, expiration_date=None)
        t["insured_objects"][0].update(object_id=None, make_model=None, year=None, location_address=None,
                                        estimated_value=None)
        t["incident_description"].update(estimated_damage_amount=None, police_report_number=None)
        self.assertEqual(fe.validate_target(t), [])


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = ft_deid.deidentify(dataset_rows())
        cls.ids, cls.examples, cls.rejected = fe.build_examples(cls.rows)

    def test_every_dataset_record_exports_cleanly(self):
        self.assertEqual(self.rejected, [])
        self.assertEqual(len(self.examples), 30)
        self.assertEqual(self.ids, [f"rec-{i:03d}" for i in range(1, 31)])

    def test_each_example_is_system_user_assistant_with_the_claim_and_the_json_answer(self):
        for r, ex in zip(self.rows, self.examples):
            self.assertEqual(list(ex), ["messages"])
            roles = [m["role"] for m in ex["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])
            self.assertEqual(ex["messages"][0]["content"], fe.SYSTEM_PROMPT)
            self.assertEqual(ex["messages"][1]["content"], r["claim_text"])
            answer = json.loads(ex["messages"][2]["content"])
            self.assertEqual(answer, r["target"])
            self.assertEqual({s: list(v) if isinstance(v, dict) else None for s, v in answer.items() if v and s != "insured_objects"},
                             {s: KEYS[s] for s in answer if answer[s] and s != "insured_objects"})

    def test_the_answer_uses_the_schema_key_order_and_no_whitespace(self):
        for ex in self.examples:
            text = ex["messages"][2]["content"]
            self.assertNotIn("\n", text)
            self.assertEqual(list(json.loads(text)), list(KEYS))
            self.assertTrue(text.startswith('{"header":{"claim_id":'))

    def test_nothing_internal_reaches_the_examples(self):
        blob = json.dumps(self.examples, ensure_ascii=False)
        for forbidden in ("rec-0", "provenance", "warnings", "review_notes", "flags", "dataset_only", "agree\""):
            self.assertNotIn(forbidden, blob, forbidden)

    def test_the_system_prompt_states_the_schema_every_allowed_value_and_the_rules(self):
        p = fe.SYSTEM_PROMPT
        for field, values in VOCAB.items():
            for v in values:
                self.assertIn(v, p, f"{field}: {v}")
        for sec, keys in KEYS.items():
            self.assertIn(f'"{sec}"', p)
            for k in keys:
                self.assertIn(f'"{k}"', p)
        for must in ("ONE JSON object", "null", "never guess", "YYYY-MM-DD", "no $ or commas", "policy_details and insured_objects are null"):
            self.assertIn(must, p)
        self.assertEqual(fe.SYSTEM_PROMPT, fe.build_system_prompt())  # deterministic

    def test_a_bad_record_is_rejected_with_reasons_and_the_rest_still_export(self):
        rows = copy.deepcopy(self.rows)
        rows[3]["target"]["incident_description"]["incident_type"] = "flood"
        ids, examples, rejected = fe.build_examples(rows)
        self.assertEqual(len(examples), 29)
        self.assertNotIn("rec-004", ids)
        self.assertEqual([r["id"] for r in rejected], ["rec-004"])
        self.assertTrue(any("incident_type" in p for p in rejected[0]["problems"]))

    def test_inputs_are_not_modified_and_unicode_survives(self):
        rows = [row(1, "I’m Priya — café", good())]
        before = copy.deepcopy(rows)
        _, examples, _ = fe.build_examples(rows)
        self.assertEqual(rows, before)
        self.assertEqual(examples[0]["messages"][1]["content"], "I’m Priya — café")


class Files(unittest.TestCase):
    def test_write_examples_is_id_free_deterministic_and_has_an_index(self):
        rows, _ = ft_deid.deidentify(dataset_rows()[:10])
        with tempfile.TemporaryDirectory() as d:
            paths, rejected = fe.write_examples(rows, d)
            self.assertEqual(rejected, [])
            lines = [json.loads(x) for x in paths["examples"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 10)
            self.assertTrue(all(list(x) == ["messages"] for x in lines))
            index = json.loads(paths["index"].read_text(encoding="utf-8"))
            self.assertEqual(index["ids"], [f"rec-{i:03d}" for i in range(1, 11)])
            self.assertEqual(index["system_prompt_sha256"], hashlib.sha256(fe.SYSTEM_PROMPT.encode()).hexdigest())
            self.assertEqual(set(index["stats"]), {"examples", "user_chars", "assistant_chars", "approx_tokens_per_example"})
            first = {k: p.read_bytes() for k, p in paths.items()}
            fe.write_examples(rows, d)
            self.assertEqual({k: p.read_bytes() for k, p in paths.items()}, first)
            self.assertNotIn("rec-0", paths["examples"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
