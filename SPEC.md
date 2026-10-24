# Product specification — insurance_llm

> Status: draft, awaiting human approval. A coding agent must not implement product code until this specification is approved through Genesis.
> Source of truth for pipeline rules: `adr-01-pipeline_decision.md` (section numbers below refer to it).

## Problem

Insurance claim reports arrive as unstructured text (email, phone transcript, portal, in-person intake). Fine-tuning and evaluating extraction models needs a consistent, validated JSON record ("golden data") per claim. Today there are 30 raw records in `insurance_claims.json` (`claim_text`, `ground_truth`) with known label inconsistencies (ADR §9) and no reproducible pipeline that produces or checks golden records.

## Users

- Primary: the owner, whose fine-tuning pipeline consumes the golden dataset.
- Secondary: annotators and a reviewer who label and adjudicate records (ADR §7).

## Functional requirements

Each pipeline step is one requirement so it can be planned, gated and proven as its own task.

- FR-1: [Step 0] A versioned schema file and controlled vocabularies (§2, §3, §4) are frozen before extraction runs; extraction reads them and rejects unknown enum values.
- FR-2: [Step 1] Ingest keeps `raw_text` unmodified, works on a copy, preserves metadata tags, fixes only unambiguous OCR, and sets `truncated: true` when text ends mid-sentence.
- FR-3: [Step 2] Extract `channel` (from tag, else inferred with `channel_inferred: true`), a normalized ISO `report_date`, and `reported_by`.
- FR-4: [Step 3] Extract `claim_id` matching `CLM-\d{6}`, preferring the one near "claim ref/number"; otherwise `null`.
- FR-5: [Step 4] Extract `policy_number` (`POL-\d{9}`, partial or garbled gives `null`), `policyholder_name`, and `coverage_type`, ignoring unrelated policies.
- FR-6: [Step 5] Resolve `effective_date` and `expiration_date` in the ADR order of precedence, using `expiration = effective + term - 1 day`.
- FR-7: [Step 6] Extract all insured objects on the policy with correct `object_id` prefixes, `year`, assembled address, and `estimated_value` only when stated as a value.
- FR-8: [Step 7] Resolve `incident_date` from explicit or relative phrases anchored to `report_date`, verifying the weekday and flagging conflicts.
- FR-9: [Step 8] Map `incident_type` and `location_type` to the controlled vocabularies; unfit cases are flagged, never given a new label.
- FR-10: [Step 9] Extract `estimated_damage_amount` as a positive integer, preferring a written exact figure, else `null`.
- FR-11: [Step 10] Extract `police_report_number` matching `PR-\d{8}-\d{4}`; none stated gives `null`.
- FR-12: [Step 11] Assemble the record and run the nine validation checks in §6, emitting flags with severity and never auto-fixing.
- FR-13: [Step 12] Write one JSONL line per record (`id`, `version`, `raw_text`, `golden`, `flags`, `review_notes`) and a flags report so a single human reviewer can fill in `review_notes` (no second annotator; §7 adjudication is dropped).
- FR-14: [Derived layer] Compute `is_active_at_incident` (true/false, or `null` when `incident_date`, `effective_date` or `expiration_date` is null) into a separate derived file; the golden schema in §2 is not changed.

## Non-functional requirements

- NFR-1: Faithfulness. A value is `null` unless stated or derivable; every non-null value must be traceable to `raw_text` (check 9 in §6).
- NFR-2: Reproducibility. Rule-based steps are deterministic; every LLM-assisted step records the model identifier and prompt version in the output.
- NFR-3: Provenance. Existing `ground_truth` is never overwritten; generated output goes to separate files.
- NFR-4: Privacy. Claim text is sent only to the configured LLM provider; no secrets or API keys are written to the repository or to traces.
- NFR-5: Each step runs standalone on a single record or the full set in under 5 minutes excluding LLM latency, with a clear error on malformed input.

## Constraints

- Python; reuse the standard library and the already-used `datasets` package before adding dependencies.
- Dataset scope: 30 records plus the reference sample (ADR §8).
- Extraction method: rules for IDs and dates, LLM for fuzzy fields, then validation and human review (decision recorded).
- LLM provider is not yet chosen (assumption recorded).

## Non-goals

- No UI or web application; command-line scripts and files only.
- No claim adjudication, coverage decision, or payout logic.
- No dataset expansion beyond the existing 30 records and the reference sample.
- No auto-fix of validation flags (ADR Step 11).
- No relabeling of ground truth by the pipeline; the seven ADR §9 known issues are out of scope and ignored.

## Acceptance criteria

- AC-1: The reference record in ADR §8 produces exactly the golden output in §8.3.
- AC-2: All 30 records produce a schema-valid record with zero Error-severity flags, or every Error flag is listed for human review.
- AC-3: Every ID matches its regex and every enum is in its vocabulary across all output records (checks 1 and 2).
- AC-4: For every record, `incident_date <= report_date` and `effective_date < expiration_date` hold, or a flag is raised.
- AC-5: Each of the 14 requirements (13 steps plus the derived layer) passes its own executable gate on the reference record and on the 30-record set.
- AC-6: Running the pipeline twice on the same input yields byte-identical output for all rule-based fields.
- AC-7: No existing `ground_truth` value is modified by any pipeline run.
- AC-8: The JSONL output has the §7 shape, and the flags report lists every record with at least one flag.
- AC-9: `is_active_at_incident` is true only when `effective_date <= incident_date <= expiration_date`, false when outside that window, and `null` when any of the three dates is null.

## Risks

- LLM hallucination of fuzzy fields; mitigated by null-over-guess, traceability check 9 and human review.
- Ground-truth labels in the existing set are inconsistent (ADR §9); accepted risk, explicitly out of scope per the owner.
- Only one human reviewer; no independent check of labels. Mitigated by validation flags and traceability check 9.
- Ambiguous term arithmetic ("1 year" as +1 year vs -1 day); mitigated by applying the §4 rule uniformly.
- Sending claim text to an external LLM provider; mitigated by NFR-4 and the choice of provider being the owner's.

## Open questions

- Which LLM provider and model will be used? (assumed: a Claude model behind one config value)
