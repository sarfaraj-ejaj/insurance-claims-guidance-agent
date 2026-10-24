"""Step 1: ingest and normalize. raw_text is never modified; all work is on a copy."""
import re
from difflib import SequenceMatcher

_DIGIT_FIX = str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0"})
# (prefix, digit count): only an exact-length token made of digits plus look-alikes is repaired
_IDS = (("CLM", 6), ("POL", 9))
_NAME_FIX = re.compile(r"\b([A-Z][a-z]+) \(([A-Z][a-z]+)\)")
_TERMINAL = ".!?\"'’”)*"
_SIGNOFF = re.compile(r"^(?:Signed: )?[A-Z][A-Za-z'’.-]*(?: [A-Z][A-Za-z'’.-]*){0,2}$")


def _fix_ids(text):
    def repl(m):
        prefix, body = m.group(1), m.group(2)
        n = dict(_IDS)[prefix]
        fixed = body.translate(_DIGIT_FIX)
        return f"{prefix}-{fixed}" if len(body) == n and fixed.isdigit() else m.group(0)
    return re.sub(r"\b(CLM|POL)-([0-9IlO|]+)(?![A-Za-z0-9])", repl, text)


def _fix_names(text):
    def repl(m):
        a, b = m.group(1), m.group(2)
        return b if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.8 else m.group(0)
    return _NAME_FIX.sub(repl, text)


def _truncated(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and set(l.strip()) != {"-"}]
    if not lines:
        return False
    last = lines[-1]
    if last.endswith(".."):  # trailing ellipsis: the speaker was cut off
        return True
    if last[-1] in _TERMINAL or (last.startswith("[") and last.endswith("]")):
        return False
    if len(lines) > 1 and _SIGNOFF.match(last):  # signature such as "Michelle"
        return False
    return True


def ingest(raw_text):
    """Return {raw_text, text, truncated}; `text` is the working copy (tags preserved)."""
    return {
        "raw_text": raw_text,
        "text": _fix_names(_fix_ids(raw_text)),
        "truncated": _truncated(raw_text),
    }
