"""Step 6: insured objects (ADR section 5, Step 6).

Auto policies -> one Vehicle per VIN (or one VIN-less vehicle); Property/Liability -> one Building.
Values are only taken from value wording (valued / appraised / insured for ...), never from damage quotes.
"""
import re

_MAKES = ("Honda|Toyota|Ford|BMW|Mercedes|Subaru|Mazda|Chevrolet|Chevy|Nissan|Hyundai|Kia|Volkswagen|VW|"
          "Jeep|Tesla|Audi|Lexus|Dodge|GMC|Ram|Volvo|Acura|Buick|Cadillac|Chrysler|Mitsubishi|Porsche")
_VIN = re.compile(r"(?<![A-Za-z0-9])VIN[:\s]*(?:VIN)?(\d{17})(?!\d)")
_YEAR = r"(?:19|20)\d\d"
_MODEL = re.compile(rf"\b((?:{_MAKES}))(?:,? the)?(?: ([A-Z][\w-]*|[A-Z]{{1,3}}-?\d+))?\b")
_VALUE = re.compile(
    r"(?:valued|valuation|value|worth(?! of)|apprais\w*|insured for|on file for|Val)\b[^$\n.]{0,60}?\$\s?([\d,]+(?:\.\d+)?)\s?(k)?"
    r"|\$\s?([\d,]+(?:\.\d+)?)\s?(k)?\)?[^.\n]{0,12}?\bvalue\b", re.I)
_STREET = re.compile(r"\bon ([A-Z][a-z]+(?: [A-Z][a-z]+)? (?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Blvd|Ct|Court|Way))"
                     r"(?:,? in |, )([A-Z][a-z]+)")
_HOUSE_NO = re.compile(r"house number[^\d\n]{0,40}(\d{2,5})\b", re.I)
_ADDR = re.compile(
    r"\b(\d{2,5} [A-Z][a-z]+(?: [A-Z][a-z]+)?(?: (?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Blvd|Boulevard|"
    r"Ct|Court|Way|Pl|Place)\b)?)\.?(?:,? (?:here )?in |, | )?([A-Z][a-z]+)?(?:,? ([A-Z]{2})\b)?")
_BUILDING_TYPES = (  # first listed that appears wins
    (r"town\s?(?:house|home)s?", "Townhouse"), (r"single[- ]fam\w*(?: home| house)?", "Single Family Home"),
    (r"retail shop", "Retail Shop"), (r"commercial office", "Commercial Office"),
    (r"residential property", "Residential Property"))
_BUILT = re.compile(rf"\b(?:built|constructed|construction)\b[^.\n]{{0,40}}?(?:\b({_YEAR})\b|[’‘']\s?(\d\d)\b)"
                    rf"|\b(?:building|house|home) is from ({_YEAR})", re.I)


def _money(num, k):
    v = float(num.replace(",", ""))
    return int(round(v * 1000)) if k else int(round(v))


def _value(text):
    """First stated value in value wording; None if the text only quotes damage."""
    m = _VALUE.search(text)
    if not m:
        return None
    return _money(m.group(1), m.group(2)) if m.group(1) else _money(m.group(3), m.group(4))


def _year2(yy):
    yy = int(yy)
    return 1900 + yy if yy >= 30 else 2000 + yy


def _vehicle(vin, before, after, lone=False):
    """One vehicle. Make/model: nearest make before its VIN, else after. Value: after the VIN first."""
    values = after + "\n" + before
    v = _value(values)
    if v is None:  # a bare '~$11,200' next to a VIN is a stated value; elsewhere it is not
        t = re.search(r"~\s?\$\s?([\d,]+(?:\.\d+)?)\s?(k)?", values)
        v = _money(t.group(1), t.group(2)) if t and vin else None
    out = {"object_id": f"VIN{vin}" if vin else None, "object_type": "Vehicle", "make_model": None,
           "year": None, "location_address": None, "estimated_value": v}
    found = [(src, m) for src in (before, after) for m in _MODEL.finditer(src)]
    if found:
        src, m = found[0] if lone or not list(_MODEL.finditer(before)) else \
            (before, list(_MODEL.finditer(before))[-1])
        uncertain = (re.search(r"maybe (?:a |an )?$", src[max(0, m.start() - 12): m.start()])
                     or re.match(rf" or (?:{_MAKES})\b", src[m.end():]))
        if not uncertain:
            out["make_model"] = " ".join(g for g in m.groups() if g)
            y = re.search(rf"\b({_YEAR})\b", src[max(0, m.start() - 6): m.end() + 25])
            out["year"] = int(y.group(1)) if y else None
    if out["year"] is None:  # bare 4-digit years elsewhere are timestamps; only an apostrophe year counts
        y = re.search(r"[’‘']\s?(\d\d)\b", before + " " + after)
        out["year"] = _year2(y.group(1)) if y else None
    return out


def _vehicles(text):
    vins = list(_VIN.finditer(text))
    if not vins:
        # ponytail: one VIN-less vehicle; several unnamed vehicles would need the LLM
        return [_vehicle(None, text, "", lone=True)]
    if len({m.group(1) for m in vins}) == 1:  # a lone vehicle: its value may sit in another sentence
        return [_vehicle(vins[0].group(1), text, "", lone=True)]
    wins = {}
    for i, m in enumerate(vins):  # each VIN owns the text up to its neighbours, +-120 chars
        lo = max(vins[i - 1].end() if i else 0, m.start() - 120)
        hi = min(vins[i + 1].start() if i + 1 < len(vins) else len(text), m.end() + 120)
        b, a = wins.setdefault(m.group(1), ([], []))
        b.append(text[lo:m.start()])
        a.append(text[m.end():hi])
    return [_vehicle(vin, "\n".join(b), "\n".join(a)) for vin, (b, a) in wins.items()]


def _building(text, coverage):
    prefix = "LIAB" if coverage == "Liability" else "PROP"
    oid = re.search(r"\b(PROP|LIAB)-(\d+)\b", text)
    if oid:
        object_id = f"{oid.group(1)}-{oid.group(2)}"
    else:
        m = re.search(r"\b(?:id|number)(?: is)?\s+(\d{6})(?!\d)", text)
        object_id = f"{prefix}-{m.group(1)}" if m else None
    kind = next((label for rx, label in _BUILDING_TYPES if re.search(rx, text, re.I)), None)
    if kind is None and re.search(r"\b(?:house|home)\b", text, re.I):
        # ponytail: bare 'house' -> the plain residential label (ADR 8.3); a type never named stays None
        kind = "Residential Property" if coverage == "Liability" else "Single Family Home"
    year = None
    b = _BUILT.search(text)
    if b:
        year = int(b.group(1) or b.group(3)) if (b.group(1) or b.group(3)) else _year2(b.group(2))
    addr = None
    a = _ADDR.search(text)
    frag, num = _STREET.search(text), _HOUSE_NO.search(text)
    if frag and num:  # number and street stated separately ('Church Street in Georgetown, house number 9600')
        addr = f"{num.group(1)} {frag.group(1)}, {frag.group(2)}"
    elif a:
        addr = a.group(1) + "".join(f", {g}" for g in a.groups()[1:] if g)
    return {"object_id": object_id, "object_type": "Building", "make_model": kind, "year": year,
            "location_address": addr, "estimated_value": _value(text)}


def extract_objects(text, coverage_type):
    """List of insured-object dicts for the policy, or None when there is no policy coverage."""
    if coverage_type is None:
        return None
    if coverage_type == "Auto":
        return _vehicles(text)
    return [_building(text, coverage_type)]
