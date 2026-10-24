"""Step 3: claim ID - CLM- plus six digits, preferring the one near a claim ref/number label."""
import re

_ID = re.compile(r"(?<![A-Za-z0-9])CLM-\d{6}(?![A-Za-z0-9])", re.I)
_LABEL = re.compile(
    r"(?:claim\s*(?:ref(?:erence)?|number|id|#)|claim:|submission id|reference number)"
    r"(?:\s+(?:is|was))?[\s:'\"’—-]*$", re.I)


def extract_claim_id(text):
    """Return the claim ID, or None when absent. Never guesses a partial ID."""
    found = list(_ID.finditer(text))
    if not found:
        return None
    for m in found:
        if _LABEL.search(text[max(0, m.start() - 40):m.start()]):
            return m.group().upper()
    return found[0].group().upper()
