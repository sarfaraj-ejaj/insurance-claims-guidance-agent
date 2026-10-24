"""Step 2: header fields - channel, report date, reporter (ADR section 5, Step 2)."""
import re
from datetime import date

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
# one alternation per supported full-year format; earliest match in a window wins
_DATE = re.compile(
    rf"(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})"
    rf"|(?P<dmy>\b\d{{1,2}}-\d{{1,2}}-\d{{4}}\b)"
    rf"|(?P<dmon>\b\d{{1,2}}-{_MON}-\d{{4}}\b)"
    rf"|(?P<mdy>\b{_MON}\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}\b)", re.I)

# explicit tags, tried in order; (channel, regex)
_TAGS = (
    ("Portal", r"portal|logged in user|submission (?:timestamp|id)"),
    ("Email", r"email received|subject:"),
    ("In-Person", r"in-person|intake|front desk"),
    ("Phone", r"phone (?:call|interview|log|transcript)|call[- ]in log|call transcript"),
)
# wording only, when no tag exists (flagged channel_inferred)
_WORDS = (
    ("In-Person", r"dropped by your office|walked in|in person|lobby"),
    ("Email", r"\bemail(?:ing)?\b"),
    ("Phone", r"\bcalling\b|\bphone call\b|\bon the phone\b"),
)
_NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+)+"
_NOT_NAME = r"(?!(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day)"  # weekdays only: month prefixes would reject Mark, Julia, Janet
_WHO = (
    rf"Reported by ({_NAME})", rf"Submitted by:? ({_NAME})", rf"Logged in user: ({_NAME})",
    rf"(?:Caller|Customer|Claimant Name|Claimant|Policyholder|Holder|Submitter|Name|From): ({_NAME})",
    rf"(?i:the name is|my name is|this is|I[’']m|I am) {_NOT_NAME}({_NAME})",
    rf"^\[[^\]]*\]\s*({_NAME}) was ",
    rf"\b({_NAME}) here\b",
)


def parse_date(s):
    """Parse one of the formats in _DATE to ISO text; day-first for numeric forms."""
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        y, m, d = map(int, s.split("-"))
    elif re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", s):
        d, m, y = map(int, s.split("-"))
    elif re.fullmatch(rf"\d{{1,2}}-{_MON}-\d{{4}}", s, re.I):
        d, mon, y = s.split("-")
        d, m, y = int(d), _MONTHS[mon[:3].lower()], int(y)
    else:
        mt = re.fullmatch(rf"({_MON})\.? (\d{{1,2}})(?:st|nd|rd|th)?,? (\d{{4}})", s, re.I)
        if not mt:
            raise ValueError(f"unsupported date: {s!r}")
        m, d, y = _MONTHS[mt.group(1)[:3].lower()], int(mt.group(2)), int(mt.group(3))
    return date(y, m, d).isoformat()


def _first_date(text):
    m = _DATE.search(text)
    return parse_date(m.group()) if m else None


def extract_channel(text):
    """Return (channel, inferred). Tag in the first/last lines wins; else wording, inferred=True."""
    window = text[:200] + "\n" + text[-120:]
    for channel, rx in _TAGS:
        if re.search(rx, window, re.I):
            return channel, False
    for channel, rx in _WORDS:
        if re.search(rx, text, re.I):
            return channel, True
    return None, True


def extract_report_date(text):
    """Report date = system timestamp: first date in the header window, else a bracketed tag."""
    d = _first_date(text[:200])
    if d:
        return d
    for tag in re.findall(r"\[[^\]]*\]", text):
        d = _first_date(tag)
        if d:
            return d
    return _first_date(text)


def extract_reported_by(text):
    for rx in _WHO:
        m = re.search(rx, text, re.M)
        if m:
            return m.group(1)
    return None


def extract_header(text):
    channel, inferred = extract_channel(text)
    out = {"channel": channel, "report_date": extract_report_date(text),
           "reported_by": extract_reported_by(text)}
    if inferred:
        out["channel_inferred"] = True
    return out
