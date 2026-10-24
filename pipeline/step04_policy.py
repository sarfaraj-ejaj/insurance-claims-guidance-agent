"""Step 4: policy number, policyholder and coverage type (ADR section 5, Step 4)."""
import re
from pipeline.step02_header import extract_reported_by

_POL = re.compile(r"(?<![A-Za-z0-9])POL-(\d{9})(?![A-Za-z0-9])", re.I)
_OTHER_PERSON = re.compile(
    r"(husband|wife|cousin|brother|sister|neighbou?r|friend|competitor|father|mother|"
    r"his own|her own|son|daughter)", re.I)
_POLICY_WORD = {"car": "Auto", "auto": "Auto", "vehicle": "Auto", "truck": "Auto",
                "home": "Property", "property": "Property", "building": "Property",
                "house": "Property", "liability": "Liability"}
_EXPLICIT = re.compile(
    r"\b(auto|car|vehicle|truck|home|property|building|house|liability)\b(?: insurance)? (?:policy|coverage)"
    r"|(?:policy|coverage|\bCoverage)\s*[:(]\s*\(?(auto|property|liability)\b", re.I)
# (coverage, regex, weight): 'liability' is rare in property/auto reports, so it weighs more
_SCORES = (
    ("Auto", r"\b(?:auto|car|cars|vehicle|truck|suv|driving|VIN)\b", 1),
    ("Liability", r"\bliability\b", 3),
    ("Property", r"\b(?:property|home|house|building|townhouse|residential|roof)\b", 1),
)
_SPOUSE_HOLDER = re.compile(
    r"\bmy (?:husband|wife|spouse|partner),? ([A-Z][a-z]+),? whose name (?:should be|is) on", re.I)


def extract_policy_number(text):
    """First valid POL-nnnnnnnnn; partial/garbled IDs give None. Skip relatives' policies if others exist."""
    found = list(_POL.finditer(text))
    if not found:
        return None
    prev, mine = 0, []
    for m in found:  # look back only to the previous ID so one relative's mention can't taint the next number
        if not _OTHER_PERSON.search(text[max(prev, m.start() - 60):m.start()]):
            mine.append(m)
        prev = m.end()
    return "POL-" + (mine or found)[0].group(1)


def extract_coverage_type(text, number):
    """Coverage from an explicit 'X policy/coverage' phrase near the number, else keyword scores."""
    pos = text.find(number)
    window = text[max(0, pos - 250): pos + 250] if pos >= 0 else text
    for scope in (window, text):
        m = _EXPLICIT.search(scope)
        if m:
            word = (m.group(1) or m.group(2)).lower()
            return _POLICY_WORD.get(word) or word.capitalize()
    score = {c: w * len(re.findall(rx, text, re.I)) for c, rx, w in _SCORES}
    best = max(score, key=score.get)
    return best if score[best] else None


def extract_policyholder(text, reported_by=None):
    """Holder is the policy owner, not necessarily the caller. Returns (name, inferred)."""
    reported_by = reported_by or extract_reported_by(text)
    m = _SPOUSE_HOLDER.search(text)
    if m and reported_by:
        # ponytail: assumes the spouse shares the reporter's surname; flagged via holder_inferred
        return f"{m.group(1)} {reported_by.split()[-1]}", True
    return reported_by, False


def extract_policy(text, reported_by=None):
    """Return the policy_details core dict, or None when no valid policy number is stated."""
    number = extract_policy_number(text)
    if number is None:
        return None
    holder, inferred = extract_policyholder(text, reported_by)
    out = {"policy_number": number, "policyholder_name": holder,
           "coverage_type": extract_coverage_type(text, number)}
    if inferred:
        out["holder_inferred"] = True
    return out
