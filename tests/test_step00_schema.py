import unittest
from pipeline import step00_schema as s


class Step00(unittest.TestCase):
    def test_version_and_counts(self):
        self.assertEqual(s.VERSION, "v1.0")
        self.assertEqual(len(s.VOCAB["incident_type"]), 14)
        self.assertEqual(len(s.VOCAB["location_type"]), 7)
        self.assertEqual(s.VOCAB["channel"], ("Email", "Phone", "Portal", "In-Person"))
        self.assertEqual(s.VOCAB["coverage_type"], ("Property", "Auto", "Liability"))
        self.assertEqual(s.VOCAB["object_type"], ("Building", "Vehicle"))

    def test_schema_sections(self):
        self.assertEqual(
            set(s.SCHEMA), {"header", "policy_details", "insured_objects", "incident_description"})
        self.assertIn("incident_type", s.SCHEMA["incident_description"])

    def test_accepts_known(self):
        for field, values in s.VOCAB.items():
            for v in values:
                self.assertEqual(s.check_enum(field, v), v)

    def test_rejects_unknown(self):
        for bad in ("flood", "storm damage", "", None, 3, "email"):
            with self.assertRaises(ValueError):
                s.check_enum("incident_type", bad)
        with self.assertRaises(ValueError):
            s.check_enum("channel", "Fax")
        with self.assertRaises(ValueError):
            s.check_enum("no_such_field", "x")

    def test_frozen(self):
        with self.assertRaises(TypeError):
            s.VOCAB["channel"] = ()
        with self.assertRaises(TypeError):
            s.SCHEMA["header"]["new"] = "x"
        with self.assertRaises(AttributeError):
            s.VOCAB["channel"].append("Fax")


if __name__ == "__main__":
    unittest.main()
