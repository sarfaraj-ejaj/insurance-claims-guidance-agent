"""Fine-tuning prep, step 3 (FR-17): consistent, format-preserving pseudonyms for personal data.

Replaces person names, street addresses (house number + street name), VINs and policy / claim / police /
object ids with realistic fakes - in the claim text and in the labels together - so the model still learns what
a name or a VIN looks like while no real value is left. Dates, amounts, years, makes and cities are kept.

Two phases: `Deid.__init__` collects every entity over the WHOLE dataset and assigns a one-to-one fake from a
seeded hash (no mapping table is ever written: a table of originals would itself be a leak); `mask_text` and
`mask_target` then apply it, through placeholders so a fake is never replaced again by a later rule.
`deidentify` also scans its own output for leftovers and for labels that masking cut loose from their text.
"""
import copy
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from pipeline.store import read_stage, write_stage

SEED = "ft-v1"

FIRST = """Amelia Benjamin Camila Dominic Elena Felix Gabriela Hector Isabel Jasper Katarina Leonard Marisol Nolan
Odessa Pedro Quinn Rosalind Sebastian Tamara Ulysses Valentina Wesley Ximena Yusuf Zara Adrian Beatrice Cormac
Delphine Emeric Fiona Gideon Harriet Ignacio Josephine Kieran Lorena Matteo Nadia Orlando Penelope Rafael Soledad
Tobias Ursula Vincent Willa Xavier Yolanda Zachary Alma Bruno Celeste Damian Esperanza Fabian Greta Hugo Inez
Joaquin Kendra Lucian Mirela Nikolai Octavia Priscilla Rodrigo Simone Thaddeus Uma Viktor Wilhelmina Alistair
Bianca Callum Dorothea Evander Freya Galen Helena Iris Julian Kaia Lazaro Mabel Nestor Opal Percival Renata Silas
Talia""".split()
LAST = """Abernathy Beaumont Calloway Dunmore Ellison Fairchild Galloway Hartwell Ingram Jorgensen Kensington Lockhart
Montgomery Northcott Ostrander Pemberton Quillen Radcliffe Sutherland Thornton Underwood Vandermeer Whitaker
Yarborough Zimmerman Ashworth Blackwood Carmichael Dalrymple Eastwood Fenwick Garrison Hollis Iverson Jefferies
Kingsley Lindqvist Mercer Nightingale Oakley Prescott Rutherford Stratton Tillman Ullman Valdez Winslow Yeager
Ainsley Bellamy Cavanaugh Devereux Emerson Frobisher Gallagher Hargrove Islington Jessup Kowalski Lancaster
Merriweather Norwood Okafor Pennington Rosario Sandoval Tennyson Upton Villanueva Wadsworth Yoshida Zeller
Aldridge Bancroft Corbin Delacroix Eldridge Fitzgerald Gresham Hawthorne Ibarra Jansen Keller Lindgren Maldonado
Nakamura Ortega Pruitt Quintero Reyes Salazar""".split()
STREETS = """Maple Oakridge Cedarwood Elmhurst Birchwood Willowbrook Aspen Hickory Juniper Sycamore Magnolia Chestnut
Laurel Spruce Foxglove Heron Meadowlark Primrose Silverlake Stonebridge Thistle Valleyview Wildflower Briarwood
Clearwater Dogwood Evergreen Fernwood Goldfinch Harborview Ironwood Jasmine Kestrel Lantern Mistral Northgate
Orchard Pinehurst Quail Rosemary Summit Tamarack Upland Vineyard Windmill""".split()

SUFFIXES = ("St", "Street", "Rd", "Road", "Ave", "Avenue", "Dr", "Drive", "Ln", "Lane", "Blvd", "Boulevard",
            "Ct", "Court", "Way", "Pl", "Place")
_SUFFIX = "(?:" + "|".join(SUFFIXES) + ")"
_REL = ("neighbou?rs?|friend|sister|brother|husband|wife|cousin|son|daughter|boss|coworker|manager|landlord|tenant|"
        "contractor|mechanic|officer|agent|adjuster|representative|nephew|niece|mother|father|aunt|uncle|colleague|"
        "assistant|clerk|receptionist|partner|roommate")
_OTHER = (  # people who are in the text but not in the labels
    re.compile(rf"\b(?:{_REL})[’']?s? ([A-Z][a-z]{{2,}})\b"),
    re.compile(r"\b(?:greeted|helped|assisted|handled|served|visited|contacted) by ([A-Z][a-z]{2,})\b"),
    re.compile(r"\b(?:Thanks|Thank you|Regards|Sincerely|Best|Cheers),?[ \t]*\n[ \t]*([A-Z][a-z]{2,})\b"),
    re.compile(r"\b(?:her|his|their|my) name (?:was|is) ([A-Z][a-z]{2,})\b"),
    re.compile(r"\bnamed ([A-Z][a-z]{2,})\b"),
    re.compile(r"\b([A-Z][a-z]{2,}) from (?:next door|across the street|upstairs|downstairs)\b"),
    # 'my employees were there - Esperanza and Jamie'
    re.compile(r"\b(?:employees|staff|coworkers|colleagues|tenants|neighbou?rs|friends|kids|children)\b[^.\n]{0,40}?"
               r"[\u2014\u2013:,-]\s*([A-Z][a-z]{2,}) and ([A-Z][a-z]{2,})\b"))
_TITLES = re.compile(r"\b(?:(?:Mr|Mrs|Ms|Miss)\.?|Dr\.)\s+([A-Z][a-z]{2,})\b")  # 'Mrs. Garland' -> a last name; 'Dr' alone is a street suffix
_NOT_NAMES = set("""The This That They There Please Thanks Hello Today Yesterday Tomorrow Monday Tuesday Wednesday Thursday
Friday Saturday Sunday January February March April May June July August September October November December
Rep Caller Agent Yes Well Sorry Okay Honestly Mr Mrs Ms Miss Dr""".split())
_CLM = re.compile(r"(?<![A-Za-z0-9])CLM-([0-9IlO|]{6})(?![A-Za-z0-9])")
_POL = re.compile(r"(?<![A-Za-z0-9])POL-([0-9IlO|]{9})(?![A-Za-z0-9])")
_PR = re.compile(r"(?<![A-Za-z0-9])PR-(\d{8})-(\d{4})(?!\d)")
_OBJ = re.compile(r"(?<![A-Za-z0-9])(PROP|LIAB)-(\d{3,})(?!\d)")
_VIN = re.compile(r"(?<!\d)\d{17}(?!\d)")
_VIN_TAIL = re.compile(r"((?:…|\.\.\.)\s?)(\d{4,6})(?!\d)")
_PH = re.compile(r"\x00(\d+)\x00")
_ALLOWED_CAPS = set(_NOT_NAMES) | set(SUFFIXES) | set("""Email Phone Portal Policy Claim Property Auto Liability Intake
Incident Details Date Name Notes Insured Coverage Effective Exp Expiration Submitted Submission Customer Claimant
Statement Description Location Time Hello Hi Assistant Representative Single Family Home Street Avenue Road Drive
Lane Boulevard Court Place Building Vehicle Honda Toyota Ford BMW Mercedes Subaru Mazda""".split())


def _hash(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big")


def _norm(digits):
    """OCR look-alikes inside an id's digits: I, l, | -> 1 and O -> 0."""
    return digits.translate(str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0"}))


def _parse_address(addr):
    """'1920 Pinecrest Ln, Toledo, OH' -> (number, street name, suffix or None, rest or None); None if unparsable."""
    street, _, rest = addr.partition(", ")
    toks = street.split()
    if len(toks) < 2 or not toks[0].isdigit():
        return None
    suffix = toks[-1] if len(toks) > 2 and toks[-1] in SUFFIXES else None
    name = " ".join(toks[1:-1] if suffix else toks[1:])
    return toks[0], name, suffix, rest or None


def _label_names(t):
    out = [t["header"].get("reported_by")]
    if t.get("policy_details"):
        out.append(t["policy_details"].get("policyholder_name"))
    return [n for n in out if n]


class Deid:
    def __init__(self, rows, seed=SEED):
        self.seed = seed
        targets, texts = [r["target"] for r in rows], [r["claim_text"] for r in rows]
        corpus = set(re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(texts)))

        firsts, lasts = set(), set()
        for t in targets:
            for n in _label_names(t):
                parts = n.split()
                firsts.add(parts[0])
                lasts.update(parts[1:])
        for text in texts:
            lasts.update(w for w in _TITLES.findall(text) if w not in _NOT_NAMES)
            for rx in _OTHER:
                for hit in rx.findall(text):
                    firsts.update(w for w in ((hit,) if isinstance(hit, str) else hit) if w not in _NOT_NAMES and w not in lasts)
        self.first = self._pick("first", firsts, FIRST, corpus)
        self.last = self._pick("last", lasts, LAST, corpus)

        self.addr = {}
        for t in targets:
            for o in t.get("insured_objects") or []:
                a = o.get("location_address")
                if a and _parse_address(a):
                    self.addr[a] = _parse_address(a)
        self.street = self._pick("street", {p[1] for p in self.addr.values()}, STREETS, corpus)

        digits = {"clm": set(), "pol": set(), "obj": set(), "vin": set(), "pr": set()}
        for text in texts:
            digits["clm"].update(_norm(m) for m in _CLM.findall(text))
            digits["pol"].update(_norm(m) for m in _POL.findall(text))
            digits["obj"].update(d for _, d in _OBJ.findall(text))
            digits["vin"].update(_VIN.findall(text))
            digits["pr"].update(f"{a}-{b}" for a, b in _PR.findall(text))
        for t in targets:
            h, p, inc = t["header"], t.get("policy_details") or {}, t["incident_description"]
            if h.get("claim_id"):
                digits["clm"].add(h["claim_id"][4:])
            if p.get("policy_number"):
                digits["pol"].add(p["policy_number"][4:])
            if inc.get("police_report_number"):
                digits["pr"].add(inc["police_report_number"][3:])
            for o in t.get("insured_objects") or []:
                oid = o.get("object_id") or ""
                if oid.startswith("VIN"):
                    digits["vin"].add(oid[3:])
                elif re.match(r"(PROP|LIAB)-\d+$", oid):
                    digits["obj"].add(oid.split("-")[1])
        houses = {p[0] for p in self.addr.values()}
        every = set().union(*digits.values(), houses) | {x.split("-")[-1] for x in digits["pr"]}
        self.digits = {k: {d: self._digits(k, d, every) for d in sorted(v)} for k, v in digits.items()}
        self.house = {d: self._digits("house", d, every) for d in sorted(houses)}
        self.bare = sorted({*self.digits["clm"], *self.digits["pol"], *self.digits["obj"]}, key=lambda d: (-len(d), d))

    # ---- fake assignment ---------------------------------------------------------------------------------
    def _pick(self, kind, originals, pool, forbidden):
        used, out = set(), {}
        for o in sorted(originals):
            start = _hash(self.seed, kind, o)
            for step in range(len(pool)):
                cand = pool[(start + step) % len(pool)]
                if cand not in used and cand not in forbidden:
                    used.add(cand)
                    out[o] = cand
                    break
            else:
                raise ValueError(f"fake {kind} pool is too small for this dataset")
        return out

    def _digits(self, kind, original, forbidden):
        tail = original.split("-")[-1]
        n = len(tail)
        taken = self.__dict__.setdefault("_taken", set())
        for c in range(1000):
            d = str(_hash(self.seed, kind, original, c)).zfill(n)[-n:]
            if d[0] != "0" and d != tail and d not in forbidden and (kind, n, d) not in taken:
                taken.add((kind, n, d))
                return d if "-" not in original else f"{original.split('-')[0]}-{d}"
        raise ValueError("could not draw a fake number")

    # ---- masking -----------------------------------------------------------------------------------------
    def mask_text(self, text):
        """-> (masked text, Counter of replacements by kind)."""
        subs, n = [], Counter()

        def put(fake, kind):
            subs.append(fake)
            n[kind] += 1
            return f"\x00{len(subs) - 1}\x00"

        t = _VIN.sub(lambda m: put(self.digits["vin"][m.group()], "vin"), text)

        def tail(m):
            for orig, fake in self.digits["vin"].items():
                if orig.endswith(m.group(2)):
                    return m.group(1) + put(fake[-len(m.group(2)):], "vin_suffix")
            return m.group()
        t = _VIN_TAIL.sub(tail, t)
        t = _CLM.sub(lambda m: "CLM-" + put(self.digits["clm"][_norm(m.group(1))], "claim_id"), t)
        t = _POL.sub(lambda m: "POL-" + put(self.digits["pol"][_norm(m.group(1))], "policy_number"), t)
        t = _PR.sub(lambda m: f"PR-{m.group(1)}-" + put(self.digits["pr"][f"{m.group(1)}-{m.group(2)}"].split("-")[-1],
                                                         "police_report"), t)
        t = _OBJ.sub(lambda m: m.group(1) + "-" + put(self.digits["obj"][m.group(2)], "object_id"), t)
        for d in self.bare:
            fake = (self.digits["clm"].get(d) or self.digits["pol"].get(d) or self.digits["obj"][d])
            t = re.sub(rf"(?<![\d$\-A-Za-z\x00])(?<!\d,){d}(?!\d|,\d)", lambda m, f=fake: put(f, "bare_id"), t)

        for _, (num, name, suffix, _rest) in sorted(self.addr.items()):
            fake_name, nm, fake_num = self.street[name], re.escape(name), self.house[num]
            t = re.sub(rf"(?<![\d$\-])(?<!\d,)\b{re.escape(num)}(\s+){nm}\b",  # '300 Market' - number and name together
                       lambda m: put(fake_num, "house_number") + m.group(1) + put(fake_name, "street"), t)
            t = re.sub(rf"\b{nm}(?=\s+{_SUFFIX}\b)", lambda m: put(fake_name, "street"), t)
            t = re.sub(rf"(\b(?:on|at|over on|off)\s+){nm}\b", lambda m: m.group(1) + put(fake_name, "street"), t)
        for num in sorted(self.house):  # a short number ('58', '300') only when it is labelled as the house number
            ctx = "" if len(num) >= 4 else r"(?:(?<=number )|(?<=number: )|(?<=# ))"
            t = re.sub(rf"{ctx}(?<![\d$\-\x00])(?<!\d,){num}(?!\d|,\d)", lambda m, f=self.house[num]: put(f, "house_number"), t)

        names = [(w, f, "person_first") for w, f in self.first.items()] + [(w, f, "person_last") for w, f in self.last.items()]
        for word, fake, kind in sorted(names, key=lambda x: (-len(x[0]), x[0])):
            t = re.sub(rf"\b{re.escape(word)}\b", lambda m, f=fake, k=kind: put(f, k), t)
        known = {**self.first, **self.last}
        for w in set(re.findall(r"\b[A-Z][a-z]{4,}\b", t)):  # a typo of a name ('Patrica') leaks the name too
            best = max((SequenceMatcher(None, w.lower(), k.lower()).ratio(), k) for k in known if len(k) >= 5)
            if w not in known and best[0] >= 0.85:
                t = re.sub(rf"\b{w}\b", lambda m, f=known[best[1]]: put(f, "typo_variant"), t)
        return _PH.sub(lambda m: subs[int(m.group(1))], t), n

    def _name(self, full):
        parts = full.split()
        return " ".join([self.first.get(parts[0], parts[0])] + [self.last.get(p, p) for p in parts[1:]])

    def mask_target(self, target):
        t = copy.deepcopy(target)
        h, p, inc = t["header"], t.get("policy_details"), t["incident_description"]
        if h.get("reported_by"):
            h["reported_by"] = self._name(h["reported_by"])
        if h.get("claim_id") and h["claim_id"][4:] in self.digits["clm"]:
            h["claim_id"] = "CLM-" + self.digits["clm"][h["claim_id"][4:]]
        if p:
            if p.get("policyholder_name"):
                p["policyholder_name"] = self._name(p["policyholder_name"])
            if p.get("policy_number") and p["policy_number"][4:] in self.digits["pol"]:
                p["policy_number"] = "POL-" + self.digits["pol"][p["policy_number"][4:]]
        if inc.get("police_report_number") and inc["police_report_number"][3:] in self.digits["pr"]:
            inc["police_report_number"] = "PR-" + self.digits["pr"][inc["police_report_number"][3:]]
        for o in t.get("insured_objects") or []:
            oid = o.get("object_id") or ""
            if oid.startswith("VIN") and oid[3:] in self.digits["vin"]:
                o["object_id"] = "VIN" + self.digits["vin"][oid[3:]]
            elif re.match(r"(PROP|LIAB)-\d+$", oid):
                o["object_id"] = oid.split("-")[0] + "-" + self.digits["obj"][oid.split("-")[1]]
            a = o.get("location_address")
            if a in self.addr:
                num, name, suffix, rest = self.addr[a]
                o["location_address"] = (f"{self.house[num]} {self.street[name]}" + (f" {suffix}" if suffix else "")
                                         + (f", {rest}" if rest else ""))
        return t

    def leaks(self, blob):
        """Kinds of original entity still present in `blob` (text + labels of one output record)."""
        found = [k for k in ("first", "last") for w in getattr(self, k) if re.search(rf"\b{re.escape(w)}\b", blob)]
        for kind in ("clm", "pol", "obj", "vin"):
            found += [kind for d in self.digits[kind] if re.search(rf"(?<![\d$])(?<!\d,)({d})(?!\d|,\d)", blob)]
        found += ["house_number" for d in self.house if len(d) >= 4 and re.search(rf"(?<![\d$])(?<!\d,){d}(?!\d|,\d)", blob)]
        found += ["street" for n in self.street if re.search(rf"\b{re.escape(n)}(?=\s+{_SUFFIX}\b)", blob)]
        found += ["police_report" for k in self.digits["pr"] if k in blob.replace("PR-", "")]
        return found


def _present(kind, value, text):
    if kind == "name":
        return all(re.search(rf"\b{re.escape(p)}\b", text) for p in value.split())
    if kind == "address":
        num, rest = value.split(" ", 1)
        return num in text and rest.split(",")[0].split()[0] in text
    return value in text or value.split("-")[-1] in text


def _label_values(t):
    out = [("name", n) for n in _label_names(t)]
    h, p, inc = t["header"], t.get("policy_details") or {}, t["incident_description"]
    out += [("id", v) for v in (h.get("claim_id"), p.get("policy_number"), inc.get("police_report_number")) if v]
    for o in t.get("insured_objects") or []:
        out += [("id", o["object_id"])] if o.get("object_id") else []
        out += [("address", o["location_address"])] if o.get("location_address") else []
    return out


def deidentify(rows, seed=SEED):
    """-> (masked rows, report). Rows are {id, claim_text, target, provenance, warnings}; inputs are not modified."""
    d = Deid(rows, seed)
    out, counts = [], Counter()
    for r in rows:
        text, n = d.mask_text(r["claim_text"])
        counts.update(n)
        out.append({"id": r["id"], "claim_text": text, "target": d.mask_target(r["target"]),
                    "provenance": copy.deepcopy(r.get("provenance", {})), "warnings": list(r.get("warnings", []))})
    leaks, induced, pre = [], [], 0
    for src, new in zip(rows, out):
        for kind in d.leaks(new["claim_text"] + "\n" + json.dumps(new["target"], ensure_ascii=False)):
            leaks.append({"record": new["id"], "kind": kind})
        for (kind, v_old), (_, v_new) in zip(_label_values(src["target"]), _label_values(new["target"])):
            before = _present(kind, v_old, src["claim_text"])
            if not before:
                pre += 1
            elif not _present(kind, v_new, new["claim_text"]):
                induced.append({"record": new["id"], "kind": kind})
                new["warnings"].append(f"deid_untraceable:{kind}")
    fakes = set(d.first.values()) | set(d.last.values()) | set(d.street.values())  # only fakes actually assigned
    cities = {p[3].split(",")[0] for p in d.addr.values() if p[3]}
    cand = Counter(w for r in out for w in re.findall(r"(?<=[a-z,;:)] )([A-Z][a-z]{2,})\b", r["claim_text"])
                   if w not in _ALLOWED_CAPS and w not in fakes and w not in cities)
    report = {"seed": seed, "records": len(rows),
              "entities": {"first_names": len(d.first), "last_names": len(d.last), "streets": len(d.street),
                           "house_numbers": len(d.house), **{k: len(v) for k, v in d.digits.items()}},
              "replacements": dict(sorted(counts.items())), "leaks": leaks, "mask_induced_untraceable": induced,
              "pre_existing_untraceable": pre, "residual_candidates": dict(sorted(cand.items()))}
    return out, report


def run(reconciled_path, out_dir, seed=SEED):
    """Mask reconciled.jsonl -> <out_dir>/deidentified.jsonl. Returns (path, report); the report holds no originals."""
    rows = list(read_stage(reconciled_path.parent, Path(reconciled_path).stem).values())
    rows.sort(key=lambda r: r["id"])
    out, report = deidentify(rows, seed)
    return write_stage(out_dir, "deidentified", out), report
