"""Fine-tuning prep, step 5 (FR-19): deterministic train / validation split, 80 / 20, no test set.

Stratified by incident_type with largest-remainder quotas:
  * a type with a single example always stays in train;
  * a type with two or more always keeps at least one example in train;
  * exact duplicate claims are dropped first (and listed), so one text can never sit on both sides.
Ordering uses a seeded hash, so the result does not depend on the order of the input, and a rerun is identical.
If too many types are singletons to reach 20% the shortfall is reported in the manifest rather than hidden.
"""
import hashlib
import json
from pathlib import Path

SEED = "ft-split-v1"
VAL_RATIO = 0.2
MIN_EXAMPLES = 5


def _h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def _incident_type(example):
    try:
        return json.loads(example["messages"][2]["content"])["incident_description"]["incident_type"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError("an example's assistant message is not the expected JSON record") from e


def split(ids, examples, seed=SEED, val_ratio=VAL_RATIO):
    """-> {'train': [(id, example)], 'validation': [...], 'manifest': {...}}."""
    if len(ids) != len(examples):
        raise ValueError(f"{len(ids)} ids but {len(examples)} examples")
    kept, dropped, seen = [], [], {}
    for i, e in sorted(zip(ids, examples), key=lambda p: p[0]):
        text = e["messages"][1]["content"]
        if text in seen:
            dropped.append({"id": i, "duplicate_of": seen[text]})
        else:
            seen[text] = i
            kept.append((i, e))
    n = len(kept)
    if n < MIN_EXAMPLES:
        raise ValueError(f"{n} examples is too few to split (need at least {MIN_EXAMPLES})")
    want = int(val_ratio * n + 0.5)

    classes = {}
    for i, e in kept:
        classes.setdefault(_incident_type(e), []).append(i)
    ideal = {t: val_ratio * len(m) for t, m in classes.items()}
    cap = {t: len(m) - 1 if len(m) >= 2 else 0 for t, m in classes.items()}
    quota = {t: min(int(ideal[t] + 1e-9), cap[t]) for t in classes}
    order = sorted(classes, key=lambda t: (-(ideal[t] - int(ideal[t] + 1e-9)), -len(classes[t]), t))
    left = want - sum(quota.values())
    while left > 0 and any(quota[t] < cap[t] for t in order):
        for t in order:
            if left > 0 and quota[t] < cap[t]:
                quota[t] += 1
                left -= 1

    val_ids = set()
    for t, members in classes.items():
        val_ids.update(sorted(members, key=lambda i: _h(seed, t, i))[:quota[t]])
    by_id = dict(kept)
    shuffled = lambda ids_: sorted(ids_, key=lambda i: _h(seed, "order", i))
    train = [(i, by_id[i]) for i in shuffled(by_id.keys() - val_ids)]
    val = [(i, by_id[i]) for i in shuffled(val_ids)]

    dist = lambda pairs: dict(sorted(_count(_incident_type(e) for _, e in pairs).items()))
    manifest = {
        "seed": seed, "ratio": {"train": round(1 - val_ratio, 4), "validation": val_ratio},
        "counts": {"total": n, "train": len(train), "validation": len(val)},
        "train_ids": sorted(i for i, _ in train), "validation_ids": sorted(i for i, _ in val),
        "class_distribution": {"train": dist(train), "validation": dist(val)},
        "single_example_types": sorted(t for t, m in classes.items() if len(m) == 1),
        "types_missing_from_validation": sorted(t for t in classes if quota[t] == 0),
        "dropped_duplicates": dropped, "shortfall": max(left, 0)}
    return {"train": train, "validation": val, "manifest": manifest}


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def write_split(out_dir, seed=SEED):
    """Read examples.jsonl + examples_index.json from out_dir; write train.jsonl, validation.jsonl, split_manifest.json."""
    out = Path(out_dir)
    examples = [json.loads(x) for x in (out / "examples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = json.loads((out / "examples_index.json").read_text(encoding="utf-8"))["ids"]
    result = split(ids, examples, seed)
    paths = {"train": out / "train.jsonl", "validation": out / "validation.jsonl", "manifest": out / "split_manifest.json"}
    for name in ("train", "validation"):
        with paths[name].open("w", encoding="utf-8", newline="\n") as f:
            for _, e in result[name]:
                f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
    paths["manifest"].write_text(json.dumps(result["manifest"], sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")
    return paths
