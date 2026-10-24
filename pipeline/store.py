"""Stage store: one deterministic JSONL file per pipeline step, e.g. data/stages/02_header.jsonl."""
import json
from pathlib import Path


def write_stage(directory, name, rows):
    """Write rows (dicts with a unique 'id') to <directory>/<name>.jsonl; same rows -> same bytes."""
    ids = [r.get("id") for r in rows]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError(f"stage {name}: every row needs a unique 'id'")
    path = Path(directory) / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def read_stage(directory, name):
    """Return {id: row} for a stage file; FileNotFoundError if that step has not run."""
    path = Path(directory) / f"{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}
