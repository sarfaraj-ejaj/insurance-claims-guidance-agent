"""Step 7: incident date (ADR section 5, Step 7).

Only sentences that describe the incident count. Precedence: explicit calendar date >
weekday + day-of-month (verified) > relative phrase (yesterday, N days ago) > bare weekday.
Lower-priority evidence that disagrees raises a flag instead of changing the answer.
"""
import re
from datetime import date, timedelta
from pipeline.step02_header import _DATE, parse_date

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = "january february march april may june july august september october november december".split()
_M = "|".join(m.capitalize() for m in _MONTHS) + r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"
_WD = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

_CUE = re.compile(r"\b(?:happen(?:ed)?|occurr?ed|incident|accident|crash(?:ed)?|collision|fire|flood|burst|\w*storm|"
                  r"break-?in|entry|fell|slipped|rear-ended|sideswiped|bump|hit|filing about|date of loss|dmg)\b", re.I)
_MD = re.compile(rf"\b({_M})\.? (\d{{1,2}})(?:st|nd|rd|th)?\b(?!,? \d{{4}})", re.I)
_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({_M})\b", re.I)
_DOM = re.compile(r"(?<!today's )(?<!today is )\bthe (\d{1,2})(?:st|nd|rd|th)\b", re.I)
_WDAY = re.compile(rf"\b({_WD})\b", re.I)
_REL = (  # (regex, days back); the earliest match in the sentence wins
    (re.compile(r"\bnight before last\b|\bday before yesterday\b", re.I), 2),
    (re.compile(r"\byesterday\b|\blast night\b|\bthe day before (?:I|we) (?:reported|called|filed)\b", re.I), 1),
    (re.compile(r"\btoday\b(?!['’]s|\s+is)|\bthis (?:morning|afternoon|evening)\b|\btonight\b", re.I), 0),
)
_AGO = re.compile(r"\b(one|two|three|four|five|six|seven|\d+) days? (?:ago|after it happened)\b", re.I)
_POLICY_DATES = re.compile(r"\b(?:expir\w*|effective|renew\w*|valid|until|till)\b", re.I)  # about the policy term
_LABEL = re.compile(r"Incident Details:\s*\n\s*Date:\s*([^\n]+)", re.I)


def _month(name):
    return next(i for i, m in enumerate(_MONTHS, 1) if m[:3] == name.lower()[:3])


def _latest(rep, month, day):
    """Most recent month/day on or before the report date, or None when the day does not exist."""
    for y in (rep.year, rep.year - 1):
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if d <= rep:
            # ponytail: a yearless date more than ~6 months back is a future date or a typo, not last year's
            return d if (rep - d).days <= 183 else None
    return None


def _from_dom(rep, dom, weekday=None):
    """Day-of-month in the report month or the one before it; prefers the one on the stated weekday."""
    cands = []
    for back in range(2):
        y, m = rep.year, rep.month - back
        while m < 1:
            y, m = y - 1, m + 12
        try:
            d = date(y, m, dom)
        except ValueError:
            continue
        if d <= rep:
            cands.append(d)
    if not cands:
        return None, False
    if weekday is not None:
        hit = [d for d in cands if d.weekday() == weekday]
        if hit:
            return hit[0], False
        return cands[0], True  # the number wins over the weekday; caller flags it
    return cands[0], False


def _sentence_candidates(s, rep):
    """-> (priority, date, flags) tuples for one incident sentence, every level that applies."""
    out = []
    s = re.sub(r"\[[^\]]*\]", " ", s)  # system tags are about the report, not the incident
    wd = _WDAY.search(s)
    wd_idx = _WEEKDAYS.index(wd.group(1).lower()) if wd else None

    full = [date.fromisoformat(parse_date(m.group())) for m in _DATE.finditer(s)]
    part = [] if full else [_latest(rep, _month(m.group(1)), int(m.group(2))) for m in _MD.finditer(s)] + \
                           [_latest(rep, _month(m.group(2)), int(m.group(1))) for m in _DM.finditer(s)]
    for d in filter(None, full + part):
        out.append((1, d, ["weekday_mismatch"] if wd_idx is not None and d.weekday() != wd_idx else []))

    dom = _DOM.search(s)
    if dom:
        d, bad = _from_dom(rep, int(dom.group(1)), wd_idx)
        if d:
            out.append((2, d, ["weekday_mismatch"] if bad else []))

    rel = sorted((m.start(), days) for rx, days in _REL for m in [rx.search(s)] if m)
    ago = _AGO.search(s)
    if ago:
        n = ago.group(1).lower()
        rel = sorted(rel + [(ago.start(), _NUM.get(n) or int(n))])
    if rel:
        out.append((3, rep - timedelta(days=rel[0][1]), []))

    if not out and wd_idx is not None:  # a bare weekday is only evidence when nothing better is said
        out.append((4, rep - timedelta(days=(rep.weekday() - wd_idx) % 7 or 7), []))
    return out


def _without_policy_clauses(s):
    """Drop clauses about the policy term ('it expires May 13th') but keep the rest of the sentence."""
    if not _POLICY_DATES.search(s):
        return s  # splitting on commas would cut 'January 1st, 1959' in two
    clauses = re.split(r"[,;—–]| and | but | because ", s)
    return ", ".join(c for c in clauses if not _POLICY_DATES.search(c))


def extract_incident_date(text, report_date):
    """Return (ISO date or None, flags). report_date is ISO; without it nothing can be anchored."""
    if not report_date:
        return None, []
    rep = date.fromisoformat(report_date)
    cands = []
    m = _LABEL.search(text)
    if m and _DATE.search(m.group(1)):
        cands.append((1, date.fromisoformat(parse_date(_DATE.search(m.group(1)).group())), []))
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = _without_policy_clauses(s)
        if _CUE.search(s):
            cands += _sentence_candidates(s, rep)
    # ponytail: an incident is not reported in the future or more than a year late; such a date is a
    # birthday, a policy term or a typo. Window is a constant, widen it if real claims need it.
    cands = [c for c in cands if rep - timedelta(days=365) <= c[1] <= rep]
    if not cands:
        return None, []
    best = min(p for p, _, _ in cands)
    chosen = next(c for c in cands if c[0] == best)
    flags = list(chosen[2])
    if any(d != chosen[1] for p, d, _ in cands if p > best):
        flags.append("incident_date_conflict")
    return chosen[1].isoformat(), flags
