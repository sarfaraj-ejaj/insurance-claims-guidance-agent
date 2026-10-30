# Insurance Claims Extraction

Turns free-text insurance claim reports (email, phone transcript, portal form, in-person intake) into **consistent, validated JSON records** — "golden data" — and then into **de-identified train / validation files** ready for supervised fine-tuning of an extraction model.

Every step is its own small module with its own tests, writes its own output file, and never guesses: a value that is not stated (or derivable) in the claim is `null`, and anything doubtful is **flagged for a human reviewer** instead of being silently fixed.

> **Status: working demo.** The 14-step extraction pipeline and the 6-step fine-tuning preparation are built and tested (236 tests, no network). The two Claude-backed steps are verified with a fake model only, and the fine-tuning set is far too small to train a useful model — see [Limitations](#limitations).

---

## Contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Architecture](#architecture)
4. [The extraction steps](#the-extraction-steps)
5. [Data layout](#data-layout)
6. [The golden record](#the-golden-record)
7. [Validation and flags](#validation-and-flags)
8. [How the LLM is used](#how-the-llm-is-used)
9. [Fine-tuning dataset preparation](#fine-tuning-dataset-preparation)
10. [Design principles](#design-principles)
11. [Measured accuracy](#measured-accuracy)
12. [Testing](#testing)
13. [Repository layout](#repository-layout)
14. [Limitations](#limitations)
15. [How this was built](#how-this-was-built)

---

## What it does

**Input** — `insurance_claims.json`: 30 claims, each `{claim_text, ground_truth}` (downloaded from the public Hugging Face dataset `Cleanlab/insurance-claims-extraction` by `get_data.py`). The extraction pipeline reads only `claim_text`; `ground_truth` is only read, never modified, and is used by the tests and the fine-tuning preparation.

**Output 1 — golden records.** One record per claim with this shape (full schema in the [ADR](adr-01-pipeline_decision.md) §2):

```json
{
  "header":               { "claim_id", "report_date", "incident_date", "reported_by", "channel" },
  "policy_details":       { "policy_number", "policyholder_name", "coverage_type", "effective_date", "expiration_date" },
  "insured_objects":      [ { "object_id", "object_type", "make_model", "year", "location_address", "estimated_value" } ],
  "incident_description": { "incident_type", "location_type", "estimated_damage_amount", "police_report_number" }
}
```

plus the flags that tell a reviewer where to look.

**Output 2 — fine-tuning files.** `train.jsonl` and `validation.jsonl` (80 / 20, no test set): chat-style examples whose names, addresses, VINs and ids are replaced by consistent fakes, produced and verified by an independent checker.

## Quick start

Requirements: Python 3 (developed on 3.14). Everything uses only the standard library, except live Claude calls (`pip install anthropic`). On macOS/Linux use `python3` where this README says `py`.

```bash
py -m pipeline              # 1. raw claims -> data/stages/ -> data/final/   (golden records)
py -m pipeline.finetune     # 2. golden records -> data/finetune/            (train / validation)
```

Without an API key every rule-based layer (header, claim ID, policy, dates, objects, police report) is complete, and the three Claude fields are `null` with an `llm_unavailable` error flag. To fill them:

```bash
pip install anthropic                 # only needed for live Claude calls
export ANTHROPIC_API_KEY=...          # PowerShell: $env:ANTHROPIC_API_KEY = "..."
py -m pipeline                        # ~60 calls to claude-opus-5-5; answers are cached
```

Options: `--raw PATH` (input file), `--out DIR` (default `data/`), `--no-llm` (skip steps 8–9; `pipeline` only).
To run only the stage files: `py -m pipeline.run_stages [raw.json] [out_dir] [--llm]`.

> Running `py -m pipeline` with a key sends the claim text to Anthropic's API. Do not use real customer data unless that is acceptable for you. `py -m pipeline.finetune` makes **no** network calls.

Reruns of both commands are **byte-identical**. `pipeline` keeps any `review_notes` a human has written in `data/final/golden.jsonl`; `pipeline.finetune` keeps your decisions in `data/review/decisions.jsonl`.

## Architecture

The design is a staged pipeline where **the files on disk are the interface between steps**. Each step reads earlier stage files, writes only its own, and can be rerun alone.

```mermaid
flowchart LR
    RAW[("insurance_claims.json\n(raw, read-only)")]

    subgraph STAGES["data/stages/  - one JSONL file per step"]
        S01["01 ingest"] --> S02["02 header"]
        S01 --> S03["03 claim_id"]
        S02 --> S04["04 policy"]
        S04 --> S05["05 policy_dates"]
        S04 --> S06["06 objects"]
        S02 --> S07["07 incident_date"]
        S01 --> S10["10 police"]
        S01 -.LLM.-> S08["08 types"]
        S01 -.LLM.-> S09["09 damage"]
    end

    RAW --> S01
    LLM(["Claude\n(structured output,\ncached)"]) -.-> S08
    LLM -.-> S09

    STAGES --> ASM["11 assemble\n+ 9 validation checks"]
    ASM --> OUT["12 write_final"]
    ASM --> DER["13 derived layer"]

    OUT --> G[("data/final/golden.jsonl")]
    OUT --> F[("data/final/flags_report.json")]
    DER --> D[("data/final/derived.jsonl")]
```

Three layers, each with one job:

| Layer | Modules | Responsibility |
|---|---|---|
| **Extraction** | `step01` … `step10` | Pure functions: `text (+ earlier results) -> fields`. No I/O, no globals, easy to test. |
| **Orchestration** | `run_stages.py`, `store.py`, `__main__.py` | Run steps in order, persist each step's rows to `data/stages/NN_name.jsonl`, read earlier stages back from disk. |
| **Assembly & output** | `step11` … `step13` | Join the stage rows into the ADR record, run the validation checks, write the final files. |

Why files between steps: you can open any stage and see exactly what that step produced; a step can be rerun or replaced without touching the others; and a rerun is a plain diff. The fine-tuning preparation follows the same idea ([below](#fine-tuning-dataset-preparation)).

## The extraction steps

Follows the written design in [`adr-01-pipeline_decision.md`](adr-01-pipeline_decision.md) (§5). Each step has its own tests in `tests/`.

| # | Stage file | Module | Method | What it extracts |
|---|---|---|---|---|
| 0 | — | `step00_schema.py` | frozen constants | Schema v1.0 and controlled vocabularies; `check_enum` rejects any unknown label |
| 1 | `01_ingest` | `step01_ingest.py` | rules | Working copy of the text (raw is never changed), fixes only *unambiguous* OCR in IDs (`POL-55432I090` → `POL-554321090`) and `Name (Name)` typos, sets `truncated` when the text ends mid-sentence |
| 2 | `02_header` | `step02_header.py` | rules | Channel (from tag, else inferred and flagged), report date (many formats), reporter |
| 3 | `03_claim_id` | `step03_claim_id.py` | rules | `CLM-nnnnnn`, preferring the one next to a "claim ref/number" label |
| 4 | `04_policy` | `step04_policy.py` | rules | `POL-nnnnnnnnn` (partial/garbled → null), policyholder (may differ from the caller), coverage type; ignores relatives' policies |
| 5 | `05_policy_dates` | `step05_policy_dates.py` | rules | Effective/expiration dates: explicit date → partial date anchored to the report date → duration (`expiration = effective + term − 1 day`) → `DD/MM/YY` format hint → null |
| 6 | `06_objects` | `step06_objects.py` | rules | Vehicles (VIN, make/model, year, value) or building (id, type, year, address, value); only value wording counts, never damage quotes |
| 7 | `07_incident_date` | `step07_incident_date.py` | rules | Explicit date > weekday + day-of-month (verified) > relative phrase ("yesterday", "two days ago") > bare weekday; conflicts are flagged |
| 8 | `08_types` | `step08_types.py` | **Claude** | `incident_type`, `location_type` from the 14 + 7 allowed labels, or `none_fit` |
| 9 | `09_damage` | `step09_damage.py` | **Claude**, verified | Repair/replacement estimate; kept only if the model's quote is really in the text and contains the number |
| 10 | `10_police` | `step10_police.py` | rules | `PR-nnnnnnnn-nnnn`, else null |
| 11 | — | `step11_validate.py` | rules | Assemble the record, run the 9 validation checks |
| 12 | `final/` | `step12_output.py` | rules | `golden.jsonl` and `flags_report.json` |
| — | `final/` | `step13_derived.py` | rules | `is_active_at_incident` (kept out of the golden schema) |

## Data layout

```
insurance_claims.json        raw input - never modified
data/
  stages/                    one deterministic JSONL file per extraction step
    01_ingest.jsonl ... 10_police.jsonl
  final/
    golden.jsonl             the finished records (one per line)
    flags_report.json        records needing review, worst first
    derived.jsonl            is_active_at_incident per claim
  llm_cache/                 saved Claude answers (hash of model+prompt+schema)

  audit/                     reports about the data (no training data)
    field_agreement.json  conflicts.jsonl  profile.json  deid_report.json  readiness_report.json
  review/                    ORIGINAL (unmasked) values + your decisions - keep private
    reconciled.jsonl  excluded.jsonl  decisions.jsonl
  finetune/                  MASKED files only - what you hand to a trainer
    train.jsonl  validation.jsonl  split_manifest.json
    deidentified.jsonl  examples.jsonl  examples_index.json
```

A stage row holds only that step's fields, keyed by a stable id:

```json
{"channel":"Email","id":"rec-001","report_date":"2024-04-22","reported_by":"Jennifer Garcia"}
```

Files are written with sorted keys, UTF-8 and LF endings, so identical input always yields identical bytes. `data/` is in `.gitignore`; the generated files committed earlier stay tracked.

## The golden record

Each line of `data/final/golden.jsonl` follows ADR §7:

```json
{"id": "rec-001", "version": "v1.0", "raw_text": "...", "golden": { ... }, "flags": [ ... ], "review_notes": ""}
```

- `raw_text` is the untouched original claim.
- `flags` is the machine-generated list (below).
- `review_notes` is **for the human reviewer** and is preserved across reruns.

`policy_details` and `insured_objects` are `null` when the claim states no valid policy number, as the ADR specifies. Until a reviewer has gone through the flagged records, treat the output as a *draft* golden set.

## Validation and flags

Step 11 runs the nine checks from ADR §6. It **flags and never auto-fixes**.

| # | Check | Severity | Flag code |
|---|---|---|---|
| 1 | IDs match their regex | Error | `bad_id` |
| 2 | Enums are in the vocabulary; required ones are not null | Error | `bad_enum`, `missing_required` |
| 3 | `incident_date <= report_date` | Error | `incident_after_report` |
| 4 | `effective_date < expiration_date` | Error | `bad_policy_window` |
| 5 | Coverage matches object type (Auto → Vehicle) | Error | `coverage_object_mismatch` |
| 6 | Amounts are positive integers | Error | `bad_amount` |
| 7 | Police-report date ≈ incident date (±3 days) | Warning | `police_date_mismatch` |
| 8 | Incident inside the policy window | Info | `incident_outside_policy_window` |
| 9 | Non-null values traceable to the text | Warning | `not_traceable` |

Steps add their own flags: `truncated`, `channel_inferred`, `holder_inferred`, `weekday_mismatch`, `incident_date_conflict` (step 7), `taxonomy_gap:*`, `invalid_label:*` (step 8), `invalid_amount`, `amount_not_in_text` (step 9), and `llm_unavailable` / `stage_missing` (Error). A flag is `{code, severity, field, message}`.

## How the LLM is used

Only for the genuinely fuzzy fields (steps 8 and 9). Everything else is deterministic rules.

- **Model:** `claude-opus-5-5` via the official `anthropic` SDK; one constant, `MODEL` in `pipeline/llm.py`.
- **Structured output:** responses are constrained with a JSON schema (`output_config.format`); the schema's enum is the frozen vocabulary plus `none_fit`.
- **Closed vocabulary:** `none_fit` becomes `null` plus a `taxonomy_gap` flag — the model can never invent a label. Anything outside the vocabulary is rejected.
- **Verified, not trusted:** for the damage amount the model must return the exact quote it used; the code checks the quote is in the claim and contains the number, otherwise the value is dropped and flagged.
- **Claim text is data:** it is passed inside `<claim>` tags with an instruction to ignore instructions found there.
- **Cached:** each answer is stored in `data/llm_cache/` under a hash of model + prompt + schema, so reruns are free and identical. Only the answer is stored — never the API key or the claim text.
- **Fails safe:** no key, refusal, truncated or malformed answer, or an API error → the field is `null` with `llm_unavailable`. The refusal fallback (`fallbacks="default"`) is enabled.
- **Provenance:** every LLM stage row records the `model` and `prompt_version`.

## Fine-tuning dataset preparation

`py -m pipeline.finetune` turns the golden records into train / validation files for **supervised fine-tuning** (instruction → JSON answer). It needs `data/final/golden.jsonl` (it runs the extraction first if that is missing) and makes no network calls.

**The idea.** The dataset's `ground_truth` labels are the training targets. The pipeline's golden output is an *independent second opinion*: wherever the two disagree, the record is held back until a person decides. `ground_truth` is only ever read.

```mermaid
flowchart LR
    G[("data/final/golden.jsonl")] --> A["1 audit"]
    R[("insurance_claims.json\nground_truth")] --> A
    A --> C["2 reconcile"]
    D[/"data/review/decisions.jsonl\n(your decisions)"/] --> C
    C -->|ready| X["3 de-identify"]
    C -->|conflict| E[("data/review/excluded.jsonl")]
    X --> P["4 export"] --> S["5 split 80/20"] --> K["6 independent checks"]
    S --> T[("train.jsonl")]
    S --> V[("validation.jsonl")]
    K --> Q[("audit/readiness_report.json")]
```

| # | Step | Module | What it does |
|---|---|---|---|
| 1 | audit | `ft_audit.py` | Compares golden vs `ground_truth` field by field (`agree` / `conflict` / `pipeline_null` / `dataset_null`); class balance; personal-data inventory |
| 2 | reconcile | `ft_reconcile.py` | One target per claim with per-field provenance. A conflict keeps the record **out** of the export until `decisions.jsonl` settles it. A missing Claude field is fine only when Claude has not run; if Claude ran and found nothing, that is a conflict |
| 3 | de-identify | `ft_deid.py` | Consistent, format-preserving pseudonyms (below) in the text **and** the labels together |
| 4 | export | `ft_export.py` | Chat-format examples, each target validated against the schema and vocabularies |
| 5 | split | `ft_split.py` | Seeded 80 / 20 train / validation, stratified by incident type, no test set |
| 6 | checks | `ft_check.py` | Independent verification of the final files; writes the readiness report |

### What a training example looks like

```json
{"messages": [
  {"role": "system",    "content": "You extract structured data from one insurance claim report. ... schema, allowed values, rules ..."},
  {"role": "user",      "content": "<the de-identified claim text>"},
  {"role": "assistant", "content": "{\"header\":{\"claim_id\":\"CLM-375976\", ... },\"incident_description\":{ ... }}"}
]}
```

The files contain only `messages` — no flags, notes, provenance or internal ids.

### De-identification

| Replaced (consistently, same fake everywhere) | Kept |
|---|---|
| Person names — in the labels, anywhere in the text, people who appear only in the text (a neighbour, "Mrs. Garland", "her name was Tina"), and name typos | Dates (relative phrases like "Tuesday the 16th" depend on them) |
| House number and street name (city stays) | Amounts, years, vehicle makes and models |
| VINs, including "ends in …83745" hints | Cities |
| Policy, claim, object and police-report numbers (a police number keeps its date part) | Enum labels |

Fakes are drawn from a seeded hash, so reruns match and **no mapping table of originals is ever written**. Fake values are never words that already occur in the data. Masking uses placeholders, so a fake can never be replaced again by a later rule. The de-identifier also scans its own output for leftovers and for labels that masking cut loose from their text, and lists capitalised words it could not classify (`residual_candidates`) for a person to glance at.

### Resolving conflicts

Records whose two label sources disagree are listed in `data/review/excluded.jsonl` and printed by the command. To bring one back, add a line to `data/review/decisions.jsonl` and rerun:

```json
{"record": "rec-008", "field": "policy_details.coverage_type", "choose": "dataset", "by": "your name", "reason": "checked the claim"}
```

`choose` is `dataset`, `pipeline`, or `value` (then add `"value": ...`). `by` and `reason` are required. A record returns only when **every** conflicting field has a decision.

### The independent checks

`ft_check.py` works from the **final files** and the **original** values, and does not ask the de-identifier whether masking worked. Nine checks: `file_shape`, `schema`, `traceability` (every label found in its text before masking still is), `personal_data` (no original name, id, VIN, house number, street or police number — and no email/phone/SSN — in any file under `data/finetune/`), `duplicates`, `split`, `exclusions`, `determinism` (a fresh rebuild is byte-identical) and `size`. The tests corrupt a good export nine different ways and require each corruption to be caught. Verdict: `pass`, `pass_with_warnings` or `fail`; exit code `0` / `0` / `1` (`2` = could not run).

### Result on the sample data

| | |
|---|---|
| Claims | 30 |
| Held back (conflicting labels) | 4 — rec-005, rec-008, rec-013, rec-015 |
| Exported | 26 → **21 train / 5 validation** |
| Replacements made | 186; 0 leftovers; 0 labels cut loose from their text |
| Verdict | `pass_with_warnings` — the warning is the dataset size |

With decisions recorded for the four held-back records the split becomes 24 / 6.

## Design principles

These were fixed before any code was written (recorded as project invariants alongside `SPEC.md`) and are enforced by tests:

1. **Never guess.** Not stated and not derivable → `null`.
2. **Never invent a label.** An unfit situation is flagged, not forced into the nearest enum.
3. **Raw text is immutable.** Work happens on a copy; `raw_text` is stored byte for byte; `ground_truth` is only read.
4. **Faithfulness.** Extract what is stated, even if the policy had expired.
5. **Flag, don't fix.** Validation reports problems; a human decides.
6. **Reproducible.** Same input → same bytes. LLM answers are cached and versioned.
7. **Documented gaps.** Where a rule cannot match the dataset's label, the test lists the record and fails if the gap ever closes unnoticed.
8. **Originals stay out of training data.** Unmasked values live only under `data/review/`; nothing under `data/finetune/` may contain one.

## Measured accuracy

Agreement of the **rule-based** steps with the dataset's `ground_truth` (30 claims; policy-dependent steps are measured on the 25 claims that state a policy number):

| Step | Match | Known misses (frozen in the tests) |
|---|---|---|
| Header (channel, report date, reporter) | 30 / 30 | — |
| Claim ID | 30 / 30 | — |
| Policy number, holder, coverage | 29 / 30 | record 7: liability coverage is only implied |
| Policy dates | 24 / 25 | record 14: label uses +1 year, the ADR rule is +1 year − 1 day (ADR §9 #4) |
| Insured objects | 22 / 25 | record 4 build year (inferred from a birthday); records 7 and 12 values differ from the stated figure (ADR §9 #2/#3) |
| Incident date | 30 / 30 | — |
| Police report number | 30 / 30 | — |

The reference record in ADR §8 assembles into **exactly** the §8.3 golden output. With ground-truth labels standing in for Claude, all 30 records assemble with **zero Error flags**.

**Steps 8 and 9 (Claude) have not been measured against live Claude** — the tests prove the wiring and the safety rules with a fake model. 25 of the 26 ground-truth damage amounts appear literally in the claim text; the exception (record 14, label 37098 vs "over $37,000") is deliberately rejected as untraceable.

These known misses are exactly the records the fine-tuning audit holds back (rec-005, rec-008, rec-013, rec-015).

## Testing

236 tests, standard library only, no network (Claude is replaced by a fake client). The full run takes about two minutes, mostly the de-identification rebuilds inside the checker tests.

```bash
py -m unittest $(ls tests/test_*.py | sed 's#/#.#; s#\.py$##')           # bash / Git Bash
```

```powershell
py -m unittest (Get-ChildItem tests/test_*.py | ForEach-Object { "tests." + $_.BaseName })   # PowerShell
```

(`unittest discover` is not used because `tests/` is a namespace package.) Shared fixtures — the ADR reference record and the 30 claims — live in `tests/test_step01_ingest.py`. Each module can be run alone, e.g. `py -m unittest tests.test_ft_deid`.

## Repository layout

```
adr-01-pipeline_decision.md  design: schema, vocabularies, rules, validation, review workflow
SPEC.md                      approved specification (FR-*, NFR-*, AC-*)
get_data.py                  downloads the dataset (needs the `datasets` package)
insurance_claims.json        the 30 raw claims
pipeline/
  __main__.py                `py -m pipeline` - raw claims -> golden records
  finetune.py                `py -m pipeline.finetune` - golden records -> train/validation
  run_stages.py  store.py    stage runner and the JSONL stage store
  llm.py                     Claude client: cache, structured output, fail-safe
  step00_schema.py ... step13_derived.py     the extraction steps
  ft_audit.py ft_reconcile.py ft_deid.py ft_export.py ft_split.py ft_check.py     fine-tuning preparation
tests/                       one test module per step, plus runner / CLI / LLM / fine-tuning tests
data/                        generated output (stages/, final/, llm_cache/, audit/, review/, finetune/)
AGENTS.md  CLAUDE.md         instructions for AI coding agents working in this repo
```

## Limitations

- **Claude steps are unmeasured.** Incident type, location type and damage amount have not been run against live Claude yet; expect to review the first output. Until then the fine-tuning targets for those fields come from the dataset labels alone, with no second opinion.
- **The fine-tuning set is far too small.** 21 training examples cannot teach a model much (extraction usually needs hundreds to thousands). Treat it as an **evaluation set** first, or add many more claims. Seven incident types have a single example (kept in train, never validated), so the 5 validation examples cover only 5 types.
- **De-identification is rule-based.** A person mentioned in a way the patterns miss stays in the text; the residual-candidate list (43 words on the sample, none of them person names) is a review aid, not a guarantee. Cities and dates are kept, public cross-street and business names (e.g. a store) are kept, and a partial id of someone else's policy (`POL-1938something`) stays. Do not treat the output as safe for real customer data without a human review.
- **Rule-based extraction is tuned to this 30-claim sample.** It generalises through broad patterns and synthetic tests, but new phrasing will produce `null` or a flag rather than a wrong value — that is by design, not a guarantee of coverage.
- **Single reviewer.** There is no second annotator, so the ADR's adjudication step is dropped; flags, traceability checks and `decisions.jsonl` are the only safety net.
- **The dataset's own label problems** (ADR §9) are ignored by the extraction pipeline; the fine-tuning audit surfaces them and holds the affected records back. Nothing ever edits `ground_truth`.
- **Privacy.** With a key set, claim text goes to Anthropic. `data/review/` holds unmasked values and the cache and stage files hold claim text — treat `data/` accordingly.
- **Staleness.** The files committed under `data/` are a snapshot from a run *without* an API key (Claude fields empty). Rerun with a key to refresh them.

## How this was built

Spec-first with the [Genesis](https://github.com/ayush488-glitch/genesis-kit) workflow: a written specification (`SPEC.md`: 20 functional / 7 non-functional requirements, 14 acceptance criteria) was approved before any code, then every step was built as its own gated task — tests first, then code, then a passing test gate. The specification was amended once to add the fine-tuning preparation, which was approved again before any of that code was written. Medium-risk tasks (the Claude client, steps 8–9 and the de-identifier) also required an independent human review. 25 tasks were completed. The workflow state lives in a local `.genesis/` folder that is intentionally not committed.
