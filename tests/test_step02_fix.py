import unittest
from pipeline.step02_header import extract_reported_by as who


class Step02Fix(unittest.TestCase):
    def test_names_that_start_with_a_month_prefix_are_kept(self):
        # a naive "fixed" month guard (Mar[a-z]*) would reject all of these
        for text, name in (("This is Mark Evans.", "Mark Evans"), ("The name is Mark Evans.", "Mark Evans"),
                           ("I'm Julia Roberts and", "Julia Roberts"), ("I\u2019m Janet Cole,", "Janet Cole"),
                           ("my name is Marcus Brown", "Marcus Brown"), ("This is Augustine Lee.", "Augustine Lee"),
                           ("I am Decker Stone", "Decker Stone")):
            self.assertEqual(who(text), name, text)

    def test_weekdays_and_dates_are_not_names(self):
        for text in ("This is Saturday September 20th", "I'm Monday Morning", "this is Sunday Evening again"):
            self.assertIsNone(who(text), text)

    def test_the_guard_source_has_no_control_characters(self):
        import pipeline.step02_header as m
        data = open(m.__file__, "rb").read()
        self.assertEqual([i for i, c in enumerate(data) if c < 32 and c not in (9, 10, 13)], [])


if __name__ == "__main__":
    unittest.main()
