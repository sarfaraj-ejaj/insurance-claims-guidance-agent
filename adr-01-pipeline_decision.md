# Claim Extraction: Raw → Golden Data Pipeline

> Version: v1.0
> Owner: <your-name>
> Last updated: <YYYY-MM-DD>
> Dataset size: 30 records (+1 reference sample)

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Target Schema](#2-target-schema)
3. [Controlled Vocabularies](#3-controlled-vocabularies)
4. [Global Rules](#4-global-rules)
5. [Pipeline Steps](#5-pipeline-steps)
6. [Validation Checks](#6-validation-checks)
7. [Review Workflow](#7-review-workflow)
8. [Reference Sample Record](#8-reference-sample-record)
9. [Known Issues in Current Dataset](#9-known-issues-in-current-dataset)
10. [Changelog](#10-changelog)

---

## 1. Purpose

Convert unstructured insurance claim reports (email, phone transcript, portal, in-person intake)
into a consistent, validated JSON record ("golden data") for evaluation and fine-tuning.

---

## 2. Target Schema

```json
{
  "header": {
    "claim_id": "string | null",
    "report_date": "YYYY-MM-DD",
    "incident_date": "YYYY-MM-DD | null",
    "reported_by": "string",
    "channel": "Email | Phone | Portal | In-Person"
  },
  "policy_details": {
    "policy_number": "string",
    "policyholder_name": "string",
    "coverage_type": "Property | Auto | Liability",
    "effective_date": "YYYY-MM-DD | null",
    "expiration_date": "YYYY-MM-DD | null"
  },
  "insured_objects": [
    {
      "object_id": "string | null",
      "object_type": "Building | Vehicle",
      "make_model": "string | null",
      "year": "integer | null",
      "location_address": "string | null",
      "estimated_value": "integer | null"
    }
  ],
  "incident_description": {
    "incident_type": "enum",
    "location_type": "enum",
    "estimated_damage_amount": "integer | null",
    "police_report_number": "string | null"
  }
}
```

- `policy_details` = `null` when no policy number is stated.
- `insured_objects` = `null` when `policy_details` is `null`.

---

## 3. Controlled Vocabularies

### 3.1 channel
| Value     | Cues                                                        |
|-----------|-------------------------------------------------------------|
| Email     | `[Email received]`, "emailing", subject line                |
| Phone     | `Phone log`, `Call transcript`, "I'm calling"               |
| Portal    | `Portal submission`, "logged in user", auto-generated form  |
| In-Person | `Intake`, `front desk`, "dropped by your office", lobby     |

### 3.2 coverage_type
`Property` · `Auto` · `Liability`

### 3.3 object_type
`Building` · `Vehicle`

### 3.4 incident_type
| Value                   | Description                              |
|-------------------------|------------------------------------------|
| vandalism               | Intentional damage by others             |
| theft_burglary          | Break-in / unauthorized entry            |
| kitchen_fire            | Cooking / grease fire                    |
| electrical_fire         | Wiring fault fire                        |
| burst_pipe_flood        | Plumbing failure water damage            |
| roof_leak               | Rain ingress through roof                |
| storm_damage            | Wind / hail / fallen tree                |
| slip_and_fall           | Person slipped on premises               |
| property_injury         | Injury on premises, cause unclear        |
| product_liability       | Product malfunction causing damage       |
| rear_end_collision      | Vehicle hit from behind                  |
| side_impact_collision   | T-bone / broadside                       |
| head_on_collision       | Front-to-front                           |
| parking_lot_collision   | Low-speed lot contact / sideswipe        |

### 3.5 location_type
`residence_interior` · `residence_exterior` · `commercial_property` · `parking_lot` ·
`residential_street` · `intersection` · `highway`

> New situation that doesn't fit? **Do not invent a label.** Log it in §9 for a taxonomy update.

---

## 4. Global Rules

| Rule            | Convention                                                         |
|-----------------|--------------------------------------------------------------------|
| Null            | Not stated AND not derivable → `null`. Never guess.                |
| Dates           | ISO `YYYY-MM-DD`                                                   |
| Missing day     | "May 2024" → `2024-05-01`                                          |
| Term → expiry   | `expiration = effective + term − 1 day`                            |
| Amounts         | Integer, no `$`, no commas                                         |
| Exact vs approx | Written exact figure beats verbal estimate                         |
| IDs             | Uppercase; `CLM-\d{6}`, `POL-\d{9}`, `PR-\d{8}-\d{4}`, VIN 17 chars |
| Object ID prefix| Building → `PROP-`, liability object → `LIAB-`, vehicle → VIN      |
| Faithfulness    | Extract what is stated, even if the policy had expired            |

---

## 5. Pipeline Steps

### Step 0: Freeze schema & vocabularies
- **Input:** —
- **Rules:** §2, §3, §4 are locked before annotation starts.
- **Output:** versioned schema file.

### Step 1: Ingest & normalize
- Keep `raw_text` untouched; work on a copy.
- Preserve metadata tags (timestamps, channel tags).
- Fix OCR only when unambiguous (`I`→`1` inside IDs, obvious name typos).
- Set `truncated: true` if text ends mid-sentence.

### Step 2: Header (channel, report date, reporter)
- Channel from tag; else infer from wording and note `channel_inferred: true`.
- Report date = system timestamp; normalize formats (`16-8-2025`, `Sun, Sep 21, 2025`).
- `reported_by` = person submitting (may differ from policyholder).

### Step 3: Claim ID
- Regex `CLM-\d{6}`; prefer the one near "claim ref/number".
- Not present → `null`.

### Step 4: Policy number, holder, coverage
- Regex `POL-\d{9}`; partial/garbled → `null`.
- Holder = policy owner, not necessarily the caller.
- Ignore relatives' / competitors' / unrelated policies.

### Step 5: Policy dates
Resolve in order:
1. Explicit full date
2. Partial date anchored to report date
3. Duration → apply term rule (§4)
4. Format hint (e.g. `DD/MM/YY`)
5. Otherwise `null`

### Step 6: Insured objects
- Include all objects on the policy, even if not involved.
- Exclude other policies / future additions.
- `year` = build/model year. Complete address from fragments.
- `estimated_value` only if stated as a value (not damage).

### Step 7: Incident date
- Anchor relative phrases to report date; verify weekday.
- Conflict → prefer explicit calendar date; flag for review.
- "today / this morning" → incident_date = report_date.

### Step 8: Incident & location type
- Map to §3.4 / §3.5; choose most specific supported label.

### Step 9: Damage amount
- Repair/replacement estimate only; exact figure preferred.
- None stated → `null`.

### Step 10: Police report number
- Regex `PR-\d{8}-\d{4}`; none / "no police" → `null`.

### Step 11: Assemble & auto-validate
- Run checks in §6. Flag, don't auto-fix.

### Step 12: Human review
- See §7.

---

## 6. Validation Checks

| # | Check                                             | Severity |
|---|---------------------------------------------------|----------|
| 1 | All IDs match regex                               | Error    |
| 2 | All enums in vocabulary                           | Error    |
| 3 | `incident_date <= report_date`                    | Error    |
| 4 | `effective_date < expiration_date`                | Error    |
| 5 | Coverage ↔ object type (Auto → Vehicle)           | Error    |
| 6 | Amounts are positive integers                     | Error    |
| 7 | Police report date part ≈ incident date           | Warning  |
| 8 | Incident within policy window                     | Info     |
| 9 | Every non-null value traceable to `raw_text`      | Warning  |

---

## 7. Review Workflow

1. Two annotators label each record independently.
2. Compute field-level agreement.
3. Third reviewer adjudicates disagreements.
4. Each disagreement → guideline update in §4 / §9.
5. Store per record: `raw_text`, `golden`, `review_notes`, `version`.

### Record storage format (JSONL, one per line)
```json
{"id": "rec-001", "version": "v1.0", "raw_text": "...", "golden": {...}, "flags": [], "review_notes": ""}
```

---

## 8. Reference Sample Record

### 8.1 Raw input
```text
[Email received: 2025-10-06]
Hi, Priya Nair here. My claim ref is CLM-318842. During Saturday's windstorm (Oct 4, around 7pm)
a big oak came down on the garage of my house at 58 Lakeshore Rd, Austin. It was built in '88 and
the county valued it around $410,500 (our system id is 774120). My property policy is POL-301928475,
started Feb 15, 2024, and runs two years. The contractor said "about 12k" but the written quote is
$11,960. No police, but the city arborist came by. My old Vespa is insured elsewhere, ignore that.
```

### 8.2 Step-by-step trace
| Step | Field(s)                      | Value                          | Reasoning                         |
|------|-------------------------------|--------------------------------|-----------------------------------|
| 2    | channel / report_date         | Email / 2025-10-06             | Email tag                         |
| 3    | claim_id                      | CLM-318842                     | Explicit                          |
| 4    | policy_number / coverage      | POL-301928475 / Property       | "property policy"                 |
| 5    | effective / expiration        | 2024-02-15 / 2026-02-14        | 2-year term, −1 day rule          |
| 6    | object                        | PROP-774120, 1988, $410,500    | Vespa excluded (other insurer)    |
| 7    | incident_date                 | 2025-10-04                     | Saturday Oct 4 verified           |
| 8    | incident / location type      | storm_damage / residence_exterior | Tree on garage in windstorm    |
| 9    | estimated_damage_amount       | 11960                          | Written quote beats "about 12k"   |
| 10   | police_report_number          | null                           | "No police"                       |

### 8.3 Golden output
```json
{
  "header": {
    "claim_id": "CLM-318842",
    "report_date": "2025-10-06",
    "incident_date": "2025-10-04",
    "reported_by": "Priya Nair",
    "channel": "Email"
  },
  "policy_details": {
    "policy_number": "POL-301928475",
    "policyholder_name": "Priya Nair",
    "coverage_type": "Property",
    "effective_date": "2024-02-15",
    "expiration_date": "2026-02-14"
  },
  "insured_objects": [
    {
      "object_id": "PROP-774120",
      "object_type": "Building",
      "make_model": "Single Family Home",
      "year": 1988,
      "location_address": "58 Lakeshore Rd, Austin",
      "estimated_value": 410500
    }
  ],
  "incident_description": {
    "incident_type": "storm_damage",
    "location_type": "residence_exterior",
    "estimated_damage_amount": 11960,
    "police_report_number": null
  }
}
```

---

## 9. Known Issues in Current Dataset

| # | Record      | Issue                                                        | Action            | Status |
|---|-------------|--------------------------------------------------------------|-------------------|--------|
| 1 | CLM-424063  | Dates 2024-04-22 vs police report PR-2025…                   | Fix typo          | Open   |
| 2 | CLM-405338  | Label 37098 not derivable from "over $37,000"                | Relabel / add src | Open   |
| 3 | CLM-699674  | Label 372760 vs text "$372,000"                              | Relabel / add src | Open   |
| 4 | Multiple    | "1 year" term: +1 year vs −1 day inconsistent                | Apply §4 rule     | Open   |
| 5 | CLM-144512  | "Couple years" vs label 3 years                              | Clarify text      | Open   |
| 6 | Multiple    | Expired policies at incident: add `is_active_at_incident` in a derived layer | Add field | Open |
| 7 | Multiple    | Channel inferred without tag (Davis records)                 | Document cues     | Open   |

---

## 10. Changelog

| Version | Date       | Change                     |
|---------|------------|----------------------------|
| v1.0    | YYYY-MM-DD | Initial guidelines         |