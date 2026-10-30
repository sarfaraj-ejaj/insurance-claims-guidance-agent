# Product specification — insurance_llm

> Status: amended 2026-10-30 (fine-tuning dataset preparation, FR-15..FR-20), awaiting re-approval. A coding agent must not implement product code until this specification is approved through Genesis.
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

Fine-tuning dataset preparation (amendment 2026-10-30). The dataset's `ground_truth` supplies the training targets; the pipeline's golden output is the independent second opinion that catches label errors.

- FR-15: [Audit] Compare the pipeline golden records with the dataset `ground_truth` field by field and write an agreement report (per-field agreement, class balance, personal-data inventory) and a conflicts file (field, both values, kind: `conflict` / `pipeline_null` / `dataset_null`) under `data/audit/`.
- FR-16: [Reconcile] Build one training target per claim with per-field provenance (`agree`, `dataset_only`, `pipeline_only`, `decision`). A record with an unresolved conflict is excluded and listed with its reason; a human decision in `data/review/decisions.jsonl` resolves a conflict.
- FR-17: [De-identify] Replace person names, street addresses, VINs and policy / claim / police / object ids with realistic fake values, consistently and format-preserving, in the claim text and the labels together; run a residual scan for personal data that the rules cannot map and report it.
- FR-18: [Export] Write fine-tuning examples as chat-style JSONL (system instruction with schema and vocabulary, user = claim text, assistant = the JSON record), validated against the schema and vocabularies, with no flags, notes or internal ids.
- FR-19: [Split] Split the examples deterministically into train and validation at 80/20, stratified by `incident_type` where possible, with no test set, and write a manifest (seed, ids, counts, class distribution).
- FR-20: [Readiness] Run independent checks on the final files, write a readiness report (pass/fail per check plus warnings), and provide one command (`py -m pipeline.finetune`) that runs FR-15 to FR-20.

## Non-functional requirements

- NFR-1: Faithfulness. A value is `null` unless stated or derivable; every non-null value must be traceable to `raw_text` (check 9 in §6).
- NFR-2: Reproducibility. Rule-based steps are deterministic; every LLM-assisted step records the model identifier and prompt version in the output.
- NFR-3: Provenance. Existing `ground_truth` is never overwritten; generated output goes to separate files.
- NFR-4: Privacy. Claim text is sent only to the configured LLM provider; no secrets or API keys are written to the repository or to traces.
- NFR-5: Each step runs standalone on a single record or the full set in under 5 minutes excluding LLM latency, with a clear error on malformed input.
- NFR-6: Privacy of the export. No original personal value (name, street address, VIN, policy / claim / police / object id) from the labels appears anywhere under `data/finetune/`.
- NFR-7: Reproducibility of the export. With the same inputs, decisions and seed, every file under `data/audit/` and `data/finetune/` is byte-identical across runs; the dataset `ground_truth` is only read.

## Constraints

- Python; reuse the standard library and the already-used `datasets` package before adding dependencies.
- Dataset scope: 30 records plus the reference sample (ADR §8).
- Extraction method: rules for IDs and dates, LLM for fuzzy fields, then validation and human review (decision recorded).
- LLM: Claude (`claude-opus-5-5`) through the `anthropic` SDK for steps 8 and 9 (decision recorded). The fine-tuning preparation needs no LLM and no network.
- The export is chat-style JSONL (system / user / assistant messages). Training the model happens elsewhere.

## Non-goals

- No UI or web application; command-line scripts and files only.
- No claim adjudication, coverage decision, or payout logic.
- No dataset expansion beyond the existing 30 records and the reference sample.
- No auto-fix of validation flags (ADR Step 11).
- No relabeling of ground truth, ever: `ground_truth` is only read. The extraction pipeline ignores the seven ADR §9 known issues; the fine-tuning audit surfaces any resulting conflict, and the export refuses to train on a conflicting record until a human decides.
- No model training, and no test set: the export has train and validation only.
- No date shifting: dates are kept, because they are not identifying on their own and relative phrases in the text ("last year", "Tuesday the 16th") depend on them.
- No new claims are generated or downloaded.

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
- AC-10: After de-identification, no original name, street address, VIN or policy / claim / police / object id from the labels occurs in any file under `data/finetune/` (NFR-6), and every replacement keeps the original id format.
- AC-11: After de-identification, every non-null label value of an exported example is still found in that example's claim text (the traceability check holds), and the same original value always maps to the same fake value.
- AC-12: `train.jsonl` and `validation.jsonl` are disjoint, together contain every exported example, and the validation share is 20% of the examples (±1 example); an incident type with a single example is in train.
- AC-13: No record with an unresolved conflict reaches `train.jsonl` or `validation.jsonl`; each is listed with its reason, and a recorded human decision brings it back.
- AC-14: The readiness report gives pass/fail for schema validity, vocabulary, traceability, personal-data leaks, duplicates, split properties and determinism, and always states the dataset size and a warning when it is too small to fine-tune well.

## Risks

- LLM hallucination of fuzzy fields; mitigated by null-over-guess, traceability check 9 and human review.
- Ground-truth labels in the existing set are inconsistent (ADR §9); accepted risk, explicitly out of scope per the owner.
- Only one human reviewer; no independent check of labels. Mitigated by validation flags and traceability check 9.
- Ambiguous term arithmetic ("1 year" as +1 year vs -1 day); mitigated by applying the §4 rule uniformly.
- Sending claim text to an external LLM provider; mitigated by NFR-4 and the choice of provider being the owner's.
- Too little data: about 24 train and 6 validation examples is far too few to fine-tune a model well, and seven of the 14 incident types have one or two claims. The export is correct but only a larger set will train usefully; the readiness report must say so.
- Residual personal data: people mentioned in the text but not in the labels (neighbours, agents) are caught only by pattern rules. Mitigated by the residual scan, which reports candidates for a human to check.
- Label noise: the dataset labels and the pipeline disagree on a few records (ADR §9). Mitigated by FR-16, which keeps conflicting records out of the export until decided.

## Open questions

- Which platform and model will consume the export, and does it need a different message format? (assumed: chat-style JSONL)
- How should the records whose two label sources conflict be decided? (default: excluded until a decision is recorded in `data/review/decisions.jsonl`)
