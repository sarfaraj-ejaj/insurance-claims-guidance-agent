import collections
import json
import random
import tempfile
import unittest
from pathlib import Path
from pipeline import ft_deid, ft_export, ft_split as fs
from tests.test_ft_deid import dataset_rows


def example(i, itype, text=None):
    answer = {"incident_description": {"incident_type": itype}}
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": text or f"claim {i}"},
                         {"role": "assistant", "content": json.dumps(answer)}]}


def synthetic(counts):
    ids, exs, n = [], [], 0
    for t, c in sorted(counts.items()):
        for _ in range(c):
            n += 1
            ids.append(f"rec-{n:03d}")
            exs.append(example(n, t))
    return ids, exs


def types_of(pairs):
    return collections.Counter(json.loads(e["messages"][2]["content"])["incident_description"]["incident_type"]
                               for _, e in pairs)


class Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, _ = ft_deid.deidentify(dataset_rows())
        cls.ids, cls.examples, _ = ft_export.build_examples(rows)
        cls.result = fs.split(cls.ids, cls.examples)

    def test_ac12_sizes_disjoint_and_complete(self):
        tr, va = self.result["train"], self.result["validation"]
        self.assertEqual((len(tr), len(va)), (24, 6))
        tid, vid = {i for i, _ in tr}, {i for i, _ in va}
        self.assertFalse(tid & vid)
        self.assertEqual(tid | vid, set(self.ids))

    def test_single_example_types_are_in_train_and_every_type_keeps_a_train_example(self):
        counts = types_of(zip(self.ids, self.examples))
        tr, va = types_of(self.result["train"]), types_of(self.result["validation"])
        self.assertEqual(sum(1 for c in counts.values() if c == 1), 6)
        for t, c in counts.items():
            if c == 1:
                self.assertEqual((tr[t], va[t]), (1, 0), t)
            self.assertGreaterEqual(tr[t], 1, t)
            self.assertEqual(tr[t] + va[t], c)

    def test_stratified_the_biggest_types_are_represented_in_validation(self):
        va = types_of(self.result["validation"])
        self.assertEqual(va["slip_and_fall"], 1)
        self.assertEqual(va["rear_end_collision"], 1)
        self.assertGreaterEqual(len(va), 5)  # six validation examples spread over (at least) five types

    def test_deterministic_and_independent_of_input_order(self):
        again = fs.split(self.ids, self.examples)
        self.assertEqual(again["train"], self.result["train"])
        pairs = list(zip(self.ids, self.examples))
        random.Random(3).shuffle(pairs)
        shuffled = fs.split([i for i, _ in pairs], [e for _, e in pairs])
        self.assertEqual({i for i, _ in shuffled["validation"]}, {i for i, _ in self.result["validation"]})
        self.assertEqual(shuffled["manifest"], self.result["manifest"])

    def test_a_different_seed_gives_a_different_split(self):
        other = fs.split(self.ids, self.examples, seed="another")
        self.assertNotEqual({i for i, _ in other["validation"]}, {i for i, _ in self.result["validation"]})
        self.assertEqual(len(other["validation"]), 6)

    def test_manifest(self):
        m = self.result["manifest"]
        self.assertEqual(m["counts"], {"total": 30, "train": 24, "validation": 6})
        self.assertEqual(m["ratio"], {"train": 0.8, "validation": 0.2})
        self.assertEqual(m["train_ids"], sorted(m["train_ids"]))
        self.assertEqual(sorted(m["train_ids"] + m["validation_ids"]), sorted(self.ids))
        self.assertEqual(sum(m["class_distribution"]["train"].values()), 24)
        self.assertEqual(sum(m["class_distribution"]["validation"].values()), 6)
        self.assertEqual(m["single_example_types"], sorted(m["single_example_types"]))
        self.assertEqual(len(m["single_example_types"]), 6)
        self.assertEqual(m["types_missing_from_validation"], sorted(t for t in types_of(zip(self.ids, self.examples))
                                                                     if t not in types_of(self.result["validation"])))
        self.assertEqual((m["dropped_duplicates"], m["shortfall"]), ([], 0))


class Properties(unittest.TestCase):
    def test_for_many_random_class_mixes_the_rules_hold(self):
        rng = random.Random(11)
        for _ in range(200):
            counts = {f"t{k}": rng.choice([1, 1, 2, 2, 3, 5, 8]) for k in range(rng.randint(2, 14))}
            ids, exs = synthetic(counts)
            n = len(ids)
            if n < 5:
                continue
            r = fs.split(ids, exs)
            va = types_of(r["validation"])
            want = int(0.2 * n + 0.5)
            self.assertEqual(len(r["validation"]) + r["manifest"]["shortfall"], want, counts)
            self.assertEqual(len(r["train"]) + len(r["validation"]), n)
            for t, c in counts.items():
                if c == 1:
                    self.assertEqual(va[t], 0, counts)
                self.assertLessEqual(va[t], max(c - 1, 0), counts)

    def test_too_many_single_example_types_leave_a_reported_shortfall(self):
        ids, exs = synthetic({f"t{k}": 1 for k in range(10)})
        r = fs.split(ids, exs)
        self.assertEqual((len(r["validation"]), r["manifest"]["shortfall"]), (0, 2))

    def test_exact_duplicate_claims_never_straddle_the_split(self):
        ids, exs = synthetic({"a": 6, "b": 6})
        exs[1]["messages"][1]["content"] = exs[0]["messages"][1]["content"]  # rec-002 duplicates rec-001
        r = fs.split(ids, exs)
        self.assertEqual(r["manifest"]["dropped_duplicates"], [{"id": "rec-002", "duplicate_of": "rec-001"}])
        self.assertEqual(len(r["train"]) + len(r["validation"]), 11)
        users = [e["messages"][1]["content"] for _, e in r["train"] + r["validation"]]
        self.assertEqual(len(users), len(set(users)))

    def test_bad_inputs_are_refused(self):
        ids, exs = synthetic({"a": 3, "b": 3})
        with self.assertRaises(ValueError):
            fs.split(ids[:-1], exs)
        with self.assertRaises(ValueError):
            fs.split(ids[:4], exs[:4])  # fewer than 5 examples
        bad = list(exs)
        bad[0] = {"messages": [{"role": "assistant", "content": "not json"}] * 3}
        with self.assertRaises(ValueError):
            fs.split(ids, bad)


class Files(unittest.TestCase):
    def test_write_split_produces_id_free_files_a_manifest_and_identical_reruns(self):
        rows, _ = ft_deid.deidentify(dataset_rows())
        with tempfile.TemporaryDirectory() as d:
            ft_export.write_examples(rows, d)
            paths = fs.write_split(d)
            self.assertEqual(sorted(p.name for p in paths.values()), ["split_manifest.json", "train.jsonl", "validation.jsonl"])
            tr = [json.loads(x) for x in paths["train"].read_text(encoding="utf-8").splitlines()]
            va = [json.loads(x) for x in paths["validation"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual((len(tr), len(va)), (24, 6))
            self.assertTrue(all(list(x) == ["messages"] for x in tr + va))
            self.assertNotIn("rec-0", paths["train"].read_text(encoding="utf-8") + paths["validation"].read_text(encoding="utf-8"))
            ex_lines = sorted((Path(d) / "examples.jsonl").read_text(encoding="utf-8").splitlines())
            out_lines = sorted(paths["train"].read_text(encoding="utf-8").splitlines()
                               + paths["validation"].read_text(encoding="utf-8").splitlines())
            self.assertEqual(out_lines, ex_lines)  # nothing lost, nothing added, nothing edited
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8")) if "manifest" in paths else None
            first = {k: p.read_bytes() for k, p in paths.items()}
            fs.write_split(d)
            self.assertEqual({k: p.read_bytes() for k, p in paths.items()}, first)


if __name__ == "__main__":
    unittest.main()
