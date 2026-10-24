"""Step 10: police report number - PR-, eight digits, dash, four digits; otherwise null."""
import re

_PR = re.compile(r"(?<![A-Za-z0-9])PR-\d{8}-\d{4}(?!\d)", re.I)
_POLICE = re.compile(r"\b(?:police|officer|report)\b", re.I)


def extract_police_report_number(text):
    """Return the report number, or None when none is stated or it is partial/garbled (never guessed)."""
    found = list(_PR.finditer(text))
    for m in found:  # with several numbers, the one worded as a police/officer report is ours
        if _POLICE.search(text[max(0, m.start() - 60):m.start()]):
            return m.group().upper()
    return found[0].group().upper() if found else None
