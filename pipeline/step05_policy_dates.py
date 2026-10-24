"""Step 5: policy effective/expiration dates (ADR section 5, Step 5).

Resolution order: explicit full date > partial date anchored to the report date >
duration (expiration = effective + term - 1 day) > format hint > null.
"""
import re
from calendar import monthrange
from datetime import date, timedelta
from pipeline.step02_header import _DATE, parse_date

_MONTHS = "january february march april may june july august september october november december".split()
_M = "|".join(m.capitalize() for m in _MONTHS) + r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"
_UNITS = "first second third fourth fifth sixth seventh eighth ninth".split()
_ORD = {w: i + 1 for i, w in enumerate(_UNITS + "tenth eleventh twelfth thirteenth fourteenth fifteenth "
                                                "sixteenth seventeenth eighteenth nineteenth".split())}
_ORD.update({"twentieth": 20, "thirtieth": 30})
for _p, _b in (("twenty", 20), ("thirty", 30)):
    _ORD.update({f"{_p}-{w}": _b + i + 1 for i, w in enumerate(_UNITS)})
_WORDS = "|".join(sorted(_ORD, key=len, reverse=True))
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a couple": 2, "a couple of": 2,
         **{str(i): i for i in range(1, 6)}}
_REL = {"last": -1, "this": 0, "next": 1}
_KINDS = ("full", "relmon", "nextmonth", "md", "my", "num", "mon", "day", "relyear", "byear")

_EFF = re.compile(r"\b(?:started|start|since|effective|activated|active|went into effect|renewed|"
                  r"took out|began|got it|got the policy|set up|purchased|bought)\b", re.I)
_EXP = re.compile(r"\b(?:expir\w*|exp|until|till|through|ends?|run out|set to renew)\b", re.I)

_TOKEN = re.compile(
    rf"(?P<full>{_DATE.pattern})"
    rf"|(?P<relmon>\b(?P<rm_rel>last|this|next) (?P<rm_m>{_M})\b(?: (?P<rm_d>\d{{1,2}}|{_WORDS})\b)?)"
    rf"|(?P<nextmonth>\bnext month\b)"
    rf"|(?P<md>\b(?P<md_m>{_M})\.? (?P<md_d>\d{{1,2}}(?:st|nd|rd|th)?|{_WORDS})\b"
    rf"(?:,? (?:I think,? )?(?P<md_y>\d{{4}}))?)"
    rf"|(?P<my>\b(?P<my_m>{_M}),? (?P<my_y>\d{{4}})\b)"
    rf"|(?P<num>\((?P<n_m>\d{{1,2}})/(?P<n_d>\d{{1,2}})\))"
    rf"|(?P<mon>(?P<mon_pre>(?:the )?end of |all of )?\b(?P<mon_m>{_M})\b)"
    rf"|(?P<day>\bthe (?P<day_d>\d{{1,2}}(?:st|nd|rd|th)|{_WORDS})\b)"
    rf"|(?P<relyear>(?:of )?\b(?P<ry_rel>last|this|next) (?:year|summer|winter|spring|fall|autumn)\b)"
    rf"|(?P<byear>\b(?P<by_y>20\d\d)\b)", re.I)
_DURATION = re.compile(
    r"\b(?:(?:for|runs?|term of|in|up to) (?P<n1>one|two|three|four|five|a couple(?: of)?|[1-5])[- ]years?(?! ago)"
    r"|(?P<n2>one|two|three|four|five|a couple(?: of)?|[1-5]) years later)", re.I)
_HINT = re.compile(r"(\d\d)-(\d\d)-(\d\d)\s*/\s*(\d\d)-(\d\d)-(\d\d)\s+(DD/MM/YY|MM/DD/YY|YY/MM/DD)")


def _month(name):
    return next(i for i, m in enumerate(_MONTHS, 1) if m[:3] == name.lower()[:3])


def _day(tok):
    return _ORD.get(tok.lower()) or int(re.sub(r"\D", "", tok))


def add_years(d, n):
    y = d.year + n
    return date(y, d.month, min(d.day, monthrange(y, d.month)[1]))


def _hint_dates(text):
    """Format hint such as '10-06-23/09-06-26 DD/MM/YY' -> (effective, expiration) or None."""
    m = _HINT.search(text)
    if not m:
        return None
    out = []
    for a, b, c in ((1, 2, 3), (4, 5, 6)):
        x, y, z = int(m.group(a)), int(m.group(b)), int(m.group(c))
        d, mo, yr = {"DD/MM/YY": (x, y, z), "MM/DD/YY": (y, x, z), "YY/MM/DD": (z, y, x)}[m.group(7)]
        out.append(date(2000 + yr, mo, d))
    return out


def _mentions(sentence):
    """Date-like mentions in one sentence as dicts: kind, span, and any of y/m/d/rel/end."""
    out = []
    for m in _TOKEN.finditer(sentence):
        k = next(g for g in _KINDS if m.group(g) is not None)
        f = {"kind": k, "span": m.span(), "y": None, "m": None, "d": None, "rel": None, "end": False}
        if k == "full":
            d = date.fromisoformat(parse_date(m.group("full")))
            f.update(y=d.year, m=d.month, d=d.day)
        elif k == "relmon":
            f.update(m=_month(m.group("rm_m")), rel=_REL[m.group("rm_rel").lower()],
                     d=_day(m.group("rm_d")) if m.group("rm_d") else None)
        elif k == "nextmonth":
            f.update(m="next")
        elif k == "md":
            f.update(m=_month(m.group("md_m")), d=_day(m.group("md_d")),
                     y=int(m.group("md_y")) if m.group("md_y") else None)
        elif k == "my":
            f.update(m=_month(m.group("my_m")), y=int(m.group("my_y")))
        elif k == "num":
            f.update(m=int(m.group("n_m")), d=int(m.group("n_d")))
        elif k == "mon":
            if m.group("mon_m") == "may":  # the verb, not the month
                continue
            f.update(m=_month(m.group("mon_m")), end=bool(m.group("mon_pre")))
        elif k == "day":
            f.update(d=_day(m.group("day_d")))
        elif k == "relyear":
            f.update(rel=_REL[m.group("ry_rel").lower()])
        else:
            f.update(y=int(m.group("by_y")))
        out.append(f)
    return out


def _roles(text):
    """Assign each mention to 'eff' or 'exp' by the nearest preceding keyword in its sentence."""
    roles = {"eff": [], "exp": []}
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        kws = sorted([(m.start(), "eff") for m in _EFF.finditer(s)] + [(m.start(), "exp") for m in _EXP.finditer(s)])
        if not kws:
            continue
        ms = _mentions(s)
        partial = [g for g in ms if g["kind"] in ("md", "mon", "num", "day", "my") and g["y"] is None]
        for f in [g for g in ms if g["kind"] == "relyear"]:  # 'last year' belongs to the nearest partial date
            if partial:
                min(partial, key=lambda g: abs(g["span"][0] - f["span"][0]))["rel"] = f["rel"]
        for f in ms:
            if f["kind"] == "relyear":
                continue
            before = [(p, r) for p, r in kws if p <= f["span"][0]]
            if not before:
                continue  # a date before any keyword in its sentence is about something else
            pos, role = before[-1]
            if f["kind"] == "byear" and not (role == "exp" and f["span"][0] - pos <= 20):
                continue  # a bare year only counts right after an expiry keyword
            roles[role].append(f)
    return roles


def _merge(ms):
    out = {"y": None, "m": None, "d": None, "rel": None, "end": False}
    for f in sorted(ms, key=lambda f: -sum(f[k] is not None for k in "ymd")):
        for k in ("y", "m", "d", "rel"):
            if out[k] is None:
                out[k] = f[k]
        out["end"] = out["end"] or f["end"]
    return out


def _resolve_eff(e, rep):
    """-> (full date or None, (year, month) or None)."""
    if not isinstance(e["m"], int):
        return None, None
    y = e["y"]
    if y is None and e["rel"] is not None and rep:
        y = rep.year + e["rel"]
    if y is None and rep:  # most recent such date not after the report date
        y = rep.year if date(rep.year, e["m"], min(e["d"] or 1, monthrange(rep.year, e["m"])[1])) <= rep else rep.year - 1
    if y is None:
        return None, None
    return (date(y, e["m"], e["d"]) if e["d"] else None), (y, e["m"])


def _resolve_exp(x, eff, eff_ym, rep, expired):
    m, y = x["m"], x["y"]
    if m == "next" and rep:
        m, y = rep.month % 12 + 1, y or rep.year + (rep.month == 12)
    if m is None and x["d"] is not None and (eff or eff_ym):
        m = eff.month if eff else eff_ym[1]
    if not isinstance(m, int) or (x["d"] is None and not x["end"]):
        return None
    if y is None and x["rel"] is not None and rep:
        y = rep.year + x["rel"]
    if y is None:
        base = eff or (date(*eff_ym, 1) if eff_ym else None)
        if base is None:
            return None
        lo = base if (expired or not rep) else max(base, rep)
        y = lo.year
        while True:  # first such date after the effective date, and not already past unless it 'expired'
            c = date(y, m, min(x["d"] or 31, monthrange(y, m)[1]))
            if c > base and c >= lo:
                break
            y += 1
    return date(y, m, min(x["d"] or 31, monthrange(y, m)[1]))


def _duration(text):
    m = _DURATION.search(text)
    return _NUMW[(m.group("n1") or m.group("n2")).lower()] if m else None


def extract_policy_dates(text, report_date):
    """Return {'effective_date', 'expiration_date'} as ISO strings or None. report_date is ISO."""
    hint = _hint_dates(text)
    if hint:
        return {"effective_date": hint[0].isoformat(), "expiration_date": hint[1].isoformat()}
    rep = date.fromisoformat(report_date) if report_date else None
    eff = exp = eff_ym = None

    full = list(_DATE.finditer(text))  # an explicit 'D1 - D2' range wins outright
    for a, b in zip(full, full[1:]):
        if text[a.end():b.start()].strip() in ("-", "–", "—", "to"):
            eff, exp = (date.fromisoformat(parse_date(m.group())) for m in (a, b))
            break

    roles = _roles(text)
    e, x = _merge(roles["eff"]), _merge(roles["exp"])
    if eff is None:
        eff, eff_ym = _resolve_eff(e, rep)
    term = _duration(text)
    day_only = x["m"] is None and x["y"] is None and x["rel"] is None
    if exp is None and not (term and day_only):  # a bare 'the 16th' is weaker evidence than a stated term
        exp = _resolve_exp(x, eff, eff_ym, rep, bool(re.search(r"\bexpired\b", text, re.I)))

    if eff is None and eff_ym:  # day-less effective date
        if exp and exp.year > eff_ym[0]:  # derive the day from the expiry, if it agrees on the month
            cand = add_years(exp + timedelta(days=1), eff_ym[0] - exp.year)
            eff = cand if cand.month == eff_ym[1] else None
        eff = eff or date(eff_ym[0], eff_ym[1], 1)  # ADR: missing day -> first of the month
    if exp is None and eff and term:
        exp = add_years(eff, term) - timedelta(days=1)
    if eff is None and exp and term:
        eff = add_years(exp + timedelta(days=1), -term)
    return {"effective_date": eff.isoformat() if eff else None,
            "expiration_date": exp.isoformat() if exp else None}
