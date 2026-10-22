import ast, json
from datasets import load_dataset

ds = load_dataset("Cleanlab/insurance-claims-extraction", split="train")

records = []
for row in ds:
    gt = row["ground_truth"]
    # The viewer showed Python-style dicts (None, single quotes), so try literal_eval first
    try:
        gt = ast.literal_eval(gt)
    except (ValueError, SyntaxError):
        gt = json.loads(gt)
    records.append({"claim_text": row["claim_text"], "ground_truth": gt})

with open("insurance_claims.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Saved {len(records)} records")