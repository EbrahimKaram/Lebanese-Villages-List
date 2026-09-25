#!/usr/bin/env python3
"""match_v6.py — rerun of the fuzzy match with a hard constraint:

Review/auto candidates may ONLY come from the same district AND mohafaza
as the English village. Regenerates FuzzyMatch/Lebanese_Villages_Matched_v6.xlsx
from the v5 workbook (v5 itself is untouched).

Community suggestions: if FuzzyMatch/suggested_matches.csv exists (written by
the website's "Suggest a match" flow), each suggestion is auto-accepted as a
definite match when both names resolve to currently-unmatched rows in the same
district+mohafaza. Earliest timestamp wins on conflicts; everything is logged
on the "Suggestions" sheet.

Compound arabic district labels (e.g. "قضاءي بعلبك والهرمل") name two districts;
those rows are filed under every district they name, so seeds and candidate
groups of either district can see them.

Subdivisions: a comma-separated arabic cell (e.g. "بسكنتا جنوبي,بسكنتا شمالي")
is expanded into one pair per arabic name. After the one-to-one auto pass, a
subdivision pass attaches still-unmatched arabic rows that share the
qualifier-stripped base (جنوبي/شمالي/شرقي/غربي/فوقا/تحتا/حي …) of a definite
match in the same district+mohafaza — e.g. الفرزل الفوقا joins
Fourzol <-> الفرزل التحتا. It also attaches 'seed name + extra words'
rows (القبيات الذوق joins El-Koubayet <-> القبيات) and short-base حي
quarters (صور حي الجامع joins Sour (Tyr) <-> صور). Each arabic subdivision
still maps to exactly one english; one english may own many arabic subdivisions.

District-prefix stripping: an english name starting with its own district's
latin name ('Tripoli Al-Tabbaneh' in Tripoli) is also tried without that
prefix, since the arabic source names city quarters without the city prefix
(التبانة). The original form is still tried too.

Transliteration: ta marbuta (ة) also renders as 'e' (in addition to 't'/'a'),
matching the Lebanese '-eh' pronunciation behind french spellings like
Tabbaneh, Kobbé, Souéka.

Usage: python3 match_v6.py [--input workbook.xlsx] [--output out.xlsx]
       [--unmatch rejected.csv] [--accept approved.csv]

Reviewer overrides: --unmatch blanks rejected v5 pairs before matching;
--accept forces approved (english, arabic, district) triples as definite
matches (both sides must be unmatched and share district+mohafaza).
"""
import argparse
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import openpyxl
from rapidfuzz import fuzz

REPO = Path(__file__).resolve().parent.parent
V5 = REPO / "FuzzyMatch" / "Grok 4.6" / "Lebanese_Villages_Improved_Matching_v5.xlsx"
OUT = REPO / "FuzzyMatch" / "Lebanese_Villages_Matched_v6.xlsx"

_ap = argparse.ArgumentParser()
_ap.add_argument("--input", default=str(V5), help="source workbook (.xlsx)")
_ap.add_argument("--output", default=str(OUT), help="output workbook (.xlsx)")
_ap.add_argument("--unmatch", default=None,
                help="CSV of reviewer-rejected matches to blank before matching "
                     "(columns: english,arabic,district)")
_ap.add_argument("--accept", default=None,
                help="CSV of reviewer-approved matches to force as definite "
                     "(columns: english,arabic,district)")
_args = _ap.parse_args()
V5 = Path(_args.input)
OUT = Path(_args.output)

# reviewer overrides: (english, arabic, district) triples whose v5 match is
# rejected — the english name is treated as unmatched and re-run through matching
UNMATCH = set()
if _args.unmatch:
    import csv
    with open(_args.unmatch, encoding="utf-8-sig") as _f:
        for _row in csv.DictReader(_f):
            UNMATCH.add((_row["english"].strip(), _row["arabic"].strip(), _row["district"].strip()))
    print(f"unmatch overrides loaded: {len(UNMATCH)}")

# reviewer approvals: (english, arabic, district) triples forced as definite
# matches — both sides must be currently unmatched and share the same
# district+mohafaza; applied after the auto pass (so they can also seed the
# subdivision pass), before review/subdivision candidates are finalized
ACCEPT = []
if _args.accept:
    import csv
    with open(_args.accept, encoding="utf-8-sig") as _f:
        for _row in csv.DictReader(_f):
            ACCEPT.append((_row["english"].strip(), _row["arabic"].strip(), _row["district"].strip()))
    print(f"accept overrides loaded: {len(ACCEPT)}")

AUTO_MIN = 88.0
REVIEW_MIN = 70.0
INF = 1e9

# ---------------- normalization ----------------

def norm_ar(s: str) -> str:
    s = s or ""
    s = re.sub(r"[\u064b-\u0652\u0670]", "", s)
    s = s.replace("\u0640", "")
    s = re.sub(r"[أإآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip()


def norm_lat(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_ar_district(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"^(قضاءي?|قرى|محافظة)\s+", "", s.strip())
    return norm_ar(s)


def split_ar_cell(s: str):
    """A v5 arabic cell may hold several subdivisions: 'a,b،c' -> [a, b, c]."""
    return [p.strip() for p in re.split(r"[,،;]", s or "") if p.strip()]


# qualifier suffixes marking a subdivision (normalized forms, with/without ال)
SUBDIV_QUALS = {
    "جنوبي", "الجنوبي", "شمالي", "الشمالي", "شرقي", "الشرقي",
    "غربي", "الغربي", "فوقا", "الفوقا", "تحتا", "التحتا",
    "وسطي", "الوسطي", "اوسط", "الاوسط", "جديده", "الجديده",
    "قديمه", "القديمه",
}


def subdiv_base(ar: str):
    """Qualifier-stripped base of an arabic village name, or None.

    'الفرزل الفوقا' -> 'الفرزل'; 'بريتال حي التين' -> 'بريتال'.
    """
    toks = norm_ar(ar).split()
    if "حي" in toks:                      # neighborhood suffix: 'X حي Y' -> 'X'
        toks = toks[:toks.index("حي")]
    while toks and toks[-1] in SUBDIV_QUALS:
        toks.pop()
    base = " ".join(toks)
    return base if base and len(base.replace(" ", "")) >= 4 else None


# ---------------- transliteration (arabic -> latin) ----------------
AR2LAT = {
    "ا": "a", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s",
    "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "a",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m",
    "ن": "n", "ه": "h", "و": "w", "ي": "y", "ى": "a", "ة": "t",
    "ء": "", "أ": "a", "إ": "a", "آ": "a", "ؤ": "w", "ئ": "y",
}

ARTICLES = {"el", "al", "ed", "ad", "ech", "ach", "ul"}
DROP_EN = ARTICLES | {
    "dahr", "dhor", "dar", "deir", "dayr", "der", "kfar", "kafr", "kfer",
    "ain", "ayn", "ein", "beit", "bet", "beyt", "bait",
    "mazraa", "mazraat", "mazraet", "kherbet", "khirbet", "kherbih",
    "qasr", "kasr", "ksar", "et", "w", "qornet", "tell", "bir", "nahr",
    "jouar", "kafer",
}
DROP_AR = {"dahr", "dhr", "dar", "deir", "dayr", "dyr", "dir", "kfar", "kafr", "kfr",
           "ain", "ayn", "beit", "bet", "byt", "bit",
           "mazraa", "mazraat", "mzrat", "mzra",
           "kherbet", "khrbt", "khrbat", "qasr", "qsr", "ksr", "w", "al"}


def transliterate(s: str, ya="y", ta="t", waw="w", thal="dh", qaf="q") -> str:
    s = re.sub(r"[\u064b-\u0652\u0670\u0640]", "", s or "")
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # drop (district) hints
    out = []
    for ch in s:
        if ch == "ي":
            out.append(ya)
        elif ch == "ة":
            out.append(ta)
        elif ch in ("و", "ؤ"):
            out.append(waw)
        elif ch == "ذ":
            out.append(thal)
        elif ch == "ق":
            out.append(qaf)
        elif ch in AR2LAT:
            out.append(AR2LAT[ch])
        elif ch.strip() == "":
            out.append(" ")
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _variants(base: str):
    """orthographic variants: french-dialect swaps, double-letter collapse,
    trailing ta-marbuta-ish vowels stripped."""
    vs = {base}
    vs.add(base.replace("ou", "u").replace("ch", "sh").replace("ei", "i"))
    vs.add(re.sub(r"(.)\1+", r"\1", base))
    vs.add(" ".join(re.sub(r"[aeh]$", "", t) for t in base.split()))
    vs.add(" ".join(re.sub(r"[aeh]$", "", t) for t in re.sub(r"(.)\1+", r"\1", base).split()))
    return {v for v in vs if v}


def norm_ar_tokens(s: str):
    """All transliteration variants -> token lists, article stripped, prefixes dropped.

    ta marbuta (ة) gets three renderings: 't' (MSA-ish), 'a', and 'e' —
    Lebanese pronounces it '-eh' (Tabbaneh, Kobbé, Souéka), which is what the
    french-style english spellings reflect.
    """
    variants = set()
    for ya in ("y", "i"):
        for ta in ("t", "a", "e"):
            for waw in ("w", "u", "ou"):
                for thal in ("dh", "z"):
                    for qaf in ("q", "k"):
                        variants.add(transliterate(s, ya, ta, waw, thal, qaf))
    out = []
    for v in variants:
        toks = v.split()
        if toks:
            # strip attached definite article: "alryhant" -> "ryhant"
            toks[0] = re.sub(r"^(al|el)(?=[a-z]{2})", "", toks[0])
            if toks[0] in ("al", "el"):
                toks = toks[1:]
        toks = [t for t in toks if t and t not in DROP_AR]
        if toks:
            out.append(" ".join(toks))
    # dedupe while keeping a list
    seen, res = set(), []
    for v in out:
        for vv in _variants(v):
            if vv and vv not in seen:
                seen.add(vv)
                res.append(vv)
    return res or [""]


def norm_en_tokens(s: str):
    v0 = norm_lat(s or "")
    v0 = re.sub(r"[^a-z0-9 ]", " ", v0)
    toks = re.sub(r"\s+", " ", v0).strip().split()
    if toks and toks[0] in ARTICLES:
        toks = toks[1:]
    toks = [re.sub(r"^(el|al|ed|ad)-", "", t) for t in toks]
    toks = [t for t in toks if t and t not in DROP_EN]
    base = " ".join(toks)
    return tuple(v for v in _variants(base) if v) or ("",)


def en_name_variants(en, dkey):
    """English name, plus a district-prefix-stripped form.

    City quarters are often named '<District> <Quarter>' in the english
    source ('Tripoli Al-Tabbaneh') while the arabic source names just the
    quarter (التبانة). Stripping a leading district-name token lets those
    match; the original form is still tried too.
    """
    en = str(en)
    out = [en]
    if dkey:
        toks = en.split()
        if len(toks) > 1 and norm_lat(toks[0]) == norm_lat(dkey):
            stripped = " ".join(toks[1:])
            if stripped:
                out.append(stripped)
    return out


def skeleton(s: str) -> str:
    s = re.sub(r"[^a-z]", "", (s or "").lower())
    s = re.sub(r"[aeiou]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _full_score(ev: str, av: str) -> float:
    # short strings: no partial-ratio inflation (WRatio), strict comparison only
    if min(len(ev), len(av)) < 6:
        return max(fuzz.ratio(ev, av), fuzz.token_set_ratio(ev, av))
    return max(fuzz.WRatio(ev, av), fuzz.token_set_ratio(ev, av))


def _skel_score(ev: str, av: str) -> float:
    se, sa = skeleton(ev), skeleton(av)
    if len(se) < 4 or len(sa) < 4:
        return 0.0
    if min(len(se), len(sa)) / max(len(se), len(sa)) < 0.6:
        return 0.0
    return fuzz.WRatio(se, sa)


def pair_score(en, ar) -> float:
    """Best fuzzy score of an english name (or list of variant names)
    against an arabic name, across all transliteration variants.

    Also tries the dedicated French 'et'-compound rule ('X et Y' <->
    arabic 'X و Y'), which scores each side of the conjunction separately.
    """
    return max(_pair_score_base(en, ar), et_compound_score(en, ar))


def et_compound_score(en, ar) -> float:
    """French 'X et Y' <-> arabic 'X و Y' compounds.

    'Nammoura et Kfar Jerif' <-> 'النموره وكفر جريف': the plain fuzzy
    score undervalues these because the conjunction tokens ('et' / 'و')
    are dropped asymmetrically. Splitting on the conjunction and scoring
    each side separately (best part alignment — arabic sometimes reverses
    the order, e.g. 'شوان والعبره' for 'El-Abri et Chouan') fixes that.
    Returns 0 when the english side has no ' et ' or the part counts differ.
    """
    ens = en if isinstance(en, (list, tuple)) else [en]
    best = 0.0
    for e in ens:
        el = norm_lat(e)
        if " et " not in f" {el} ":
            continue
        en_parts = [p.strip(" -") for p in re.split(r"\set\s", el)]
        en_parts = [p for p in en_parts if p]
        # conjunction-و only: preceded by a space and attached to the next
        # word (' وكفر'). Word-internal و ('نموره') and word-initial و
        # ('وادي', followed by a space) are left alone.
        ar_parts = [p.strip(" -") for p in re.split(r"(?<=\s)\u0648(?=\S)", norm_ar(ar))]
        ar_parts = [p for p in ar_parts if p]
        if len(en_parts) < 2 or len(en_parts) != len(ar_parts):
            continue
        for perm in itertools.permutations(range(len(ar_parts))):
            s = sum(_pair_score_base(ep, ar_parts[perm[i]])
                    for i, ep in enumerate(en_parts)) / len(en_parts)
            if s > best:
                best = s
    return best


def _pair_score_base(en, ar) -> float:
    """Best fuzzy score of an english name (or list of variant names)
    against an arabic name, across all transliteration variants."""
    ens = en if isinstance(en, (list, tuple)) else [en]
    best = 0.0
    for e in ens:
        for ev in norm_en_tokens(e):
            if not ev:
                continue
            for av in norm_ar_tokens(ar):
                if not av:
                    continue
                s = _full_score(ev, av)
                if s > best:
                    best = s
                s2 = _skel_score(ev, av)
                if s2 > best:
                    best = s2
    return best


# ---------------- load v5 ----------------
wb5 = openpyxl.load_workbook(V5, read_only=True, data_only=True)

dist_rows = list(wb5["DistrictTranslation"].iter_rows(values_only=True))[1:]
lat2ar_d, ar2lat_d = {}, {}
ar_d_clean_to_lats = defaultdict(set)
for latin, arabic, _m in dist_rows:
    if latin and arabic:
        lat2ar_d[norm_lat(latin)] = arabic
        ar2lat_d[norm_ar(arabic)] = latin
        ar_d_clean_to_lats[clean_ar_district(arabic)].add(latin)
KNOWN_AR_D = list(ar2lat_d.keys())

moh_rows = list(wb5["Mohafaza Translations"].iter_rows(values_only=True))[1:]
ar2lat_m, lat2ar_m = {}, {}
for arabic, latin in moh_rows:
    if arabic and latin:
        ar2lat_m[norm_ar(arabic)] = latin
        lat2ar_m[norm_lat(latin)] = arabic


# exact aliases for Arabic district labels missing from DistrictTranslation
# (keys are clean_ar_district() output; checked before the table + fuzzy fallback)
AR_DISTRICT_ALIASES = {
    norm_ar("الضنية"): "Minieh-Danieh",
    norm_ar("صيدا"): "Saida",  # "قرى صيدا" is cleaned to "صيدا"
    norm_ar("بيروت الأولى"): "Beirut",
    norm_ar("بيروت الثانية"): "Beirut",
}


def dkey_en(latin):
    k = norm_lat(latin)
    return latin if k in lat2ar_d else None


def dkeys_ar(arabic):
    """Every latin district key an arabic district label maps to.

    Compound labels (e.g. 'قضاءي بعلبك والهرمل') map to each district they name;
    ordinary labels behave exactly like dkey_ar (single-element set).
    """
    c = clean_ar_district(arabic)
    if c in AR_DISTRICT_ALIASES:
        return {AR_DISTRICT_ALIASES[c]}
    if c in ar_d_clean_to_lats:
        return set(ar_d_clean_to_lats[c])
    best, bs = None, 0.0
    for known in KNOWN_AR_D:
        s = SequenceMatcher(None, c, known).ratio()
        if s > bs:
            best, bs = known, s
    return {ar2lat_d[best]} if bs >= 0.8 else set()


def dkey_ar(arabic):
    c = clean_ar_district(arabic)
    if c in AR_DISTRICT_ALIASES:
        return AR_DISTRICT_ALIASES[c]
    if c in ar2lat_d:
        return ar2lat_d[c]
    best, bs = None, 0.0
    for known in KNOWN_AR_D:
        s = SequenceMatcher(None, c, known).ratio()
        if s > bs:
            best, bs = known, s
    return ar2lat_d[best] if bs >= 0.8 else None


def mkey_en(latin):
    return latin if norm_lat(latin) in lat2ar_m else None


def mkey_ar(arabic):
    return ar2lat_m.get(norm_ar(arabic))


def is_placeholder(name):
    return re.fullmatch(r"n\.a\.\s*(\(\d+\))?", (name or "").strip().lower()) is not None


en_rows = list(wb5["English (Improved)"].iter_rows(values_only=True))[1:]
ar_rows = list(wb5["Arabic (Improved)"].iter_rows(values_only=True))[1:]

# already matched pairs (kept as-is); multi-arabic cells are expanded so each
# arabic subdivision is its own pair (one english may own many subdivisions)
already = []          # dicts for English (Full)
unm_en, unm_ar = [], []
subdiv_groups_v5 = []  # (en, d, m, [arabic names]) from comma-separated cells

for r in en_rows:
    en, ar, d, m, src = r[0], r[1], r[2], r[3], r[4]
    if not en or not str(en).strip() or is_placeholder(en):
        continue
    if ar and str(ar).strip():
        if (str(en).strip(), str(ar).strip(), str(d or "").strip()) in UNMATCH:
            print(f"  unmatch override: {en} <-> {ar} [{d}] -> treated as unmatched")
            ar = None
    if ar and str(ar).strip():
        parts = split_ar_cell(ar)
        if len(parts) > 1:
            subdiv_groups_v5.append((en, d, m, parts))
        for a in parts:
            already.append({"en": en, "ar": a, "d": d, "m": m, "src": src,
                            "subdiv": len(parts) > 1})
    else:
        rec = {"en": en, "ar": ar, "d": d, "m": m, "src": src}
        rec["dkey"] = dkey_en(d)
        rec["mkey"] = mkey_en(m)
        unm_en.append(rec)

ar_full = []
# English names holding a definite v5 pair after --unmatch blanking. An arabic
# row whose v5 english hint names one of these is already paired; any other
# hint (or none) means the pair did not survive, so the arabic row is unmatched
# and re-enters the pool. Without this, blanking the EN side of a rejected
# pair orphans its arabic row: excluded from both the matched and unmatched
# pools, invisible to the matcher and to every output sheet.
matched_en_names = {str(a["en"]).strip() for a in already
                    if a.get("en") and str(a["en"]).strip()}
orphan_restored = 0
for r in ar_rows:
    ar, en, d, m = r[0], r[1], r[2], r[3]
    if not ar or not str(ar).strip():
        continue
    rec = {"ar": ar, "en": en, "d": d, "m": m}
    ar_full.append(rec)
    en_hint = str(en).strip() if en and str(en).strip() else ""
    if not en_hint or en_hint not in matched_en_names:
        if en_hint:
            orphan_restored += 1
            print(f"  orphan restored: {ar} [{d}] (v5 english '{en_hint}' has no surviving pair)")
        rec["dkey"] = dkey_ar(d)
        rec["mkey"] = mkey_ar(m)
        unm_ar.append(rec)
print(f"orphaned arabic rows restored to the unmatched pool: {orphan_restored}")

print(f"already matched: {len(already)} | unmatched EN: {len(unm_en)} | unmatched AR: {len(unm_ar)}")
no_key_en = sum(1 for u in unm_en if not u["dkey"] or not u["mkey"])
no_key_ar = sum(1 for u in unm_ar if not u["dkey"] or not u["mkey"])
print(f"rows without district/mohafaza key (no candidates possible): EN={no_key_en} AR={no_key_ar}")

# ---------------- one-to-one: an arabic village already matched in the same
# district+mohafaza is NOT unmatched (same-district homonyms stay untouched) --
already_triples = defaultdict(list)   # (norm_ar, dkey, mkey) -> [english names]
for rec in already:
    if rec["ar"] and str(rec["ar"]).strip():
        t = (norm_ar(rec["ar"]), dkey_en(rec["d"]), mkey_en(rec["m"]))
        already_triples[t].append(str(rec["en"]).strip())

dup_arabic = []    # (ar, district, mohafaza, [english]) dropped from unm_ar
kept_ar = []
for u in unm_ar:
    # compound district labels count as every district they name: an arabic
    # row that duplicates a definite match in any of those districts is dropped
    hit = None
    mk = u.get("mkey")
    if mk:
        for dk in dkeys_ar(u["d"]):
            t = (norm_ar(u["ar"]), dk, mk)
            if t in already_triples:
                hit = t
                break
    if hit:
        dup_arabic.append((u["ar"], u["d"], u["m"], sorted(set(already_triples[hit]))))
    else:
        kept_ar.append(u)
unm_ar = kept_ar
print(f"one-to-one: dropped {len(dup_arabic)} arabic rows already matched in the same district+mohafaza")

conflicts = {t: sorted(set(v)) for t, v in already_triples.items()
             if len(set(v)) > 1 and t[1] and t[2]}
print(f"one-to-one conflicts (same arabic+district claimed by 2+ english): {len(conflicts)}")
for t, v in sorted(conflicts.items()):
    print(f"  CONFLICT {t[0]} [{t[1]}/{t[2]}] <-> {v}")

# ---------------- group by (dkey, mkey) ----------------
groups_en, groups_ar = defaultdict(list), defaultdict(list)
for i, u in enumerate(unm_en):
    if u["dkey"] and u["mkey"]:
        groups_en[(u["dkey"], u["mkey"])].append(i)
for j, u in enumerate(unm_ar):
    if u["mkey"]:
        # compound district labels file under every district they name
        for dk in dkeys_ar(u["d"]):
            groups_ar[(dk, u["mkey"])].append(j)

# ---------------- score + hungarian per group ----------------
auto = {}            # en_idx -> (ar_idx, score)
ar_taken = set()
review = defaultdict(list)   # en_idx -> [(ar_idx, score)]

import numpy as np  # noqa

for key in sorted(set(groups_en) & set(groups_ar)):
    eis, ajs = groups_en[key], groups_ar[key]
    n, m_ = len(eis), len(ajs)
    S = np.zeros((n, m_))
    en_variants = [en_name_variants(unm_en[ei]["en"], unm_en[ei]["dkey"]) for ei in eis]
    for ii, ei in enumerate(eis):
        eu = unm_en[ei]
        eu_key = (str(eu["en"]).strip(), str(eu["d"] or "").strip())
        for jj, aj in enumerate(ajs):
            au = unm_ar[aj]
            # a rejected (--unmatch) pair stays rejected: it can neither
            # re-match nor re-surface as a review candidate
            if (eu_key[0], str(au["ar"]).strip(), eu_key[1]) in UNMATCH:
                S[ii, jj] = -1.0
                continue
            S[ii, jj] = pair_score(en_variants[ii], au["ar"])
    # mutual-best auto matching: each side must be the other's top choice
    en_best = np.argmax(S, axis=1)   # for each en row -> best ar col
    ar_best = np.argmax(S, axis=0)   # for each ar col -> best en row
    for ii, ei in enumerate(eis):
        jj = int(en_best[ii])
        s = float(S[ii, jj])
        if s >= AUTO_MIN and int(ar_best[jj]) == ii and ajs[jj] not in ar_taken:
            auto[ei] = (ajs[jj], s)
            ar_taken.add(ajs[jj])
    # review: top-3 per en (excluding auto-taken ar), score in [REVIEW_MIN, AUTO_MIN)
    for ii, ei in enumerate(eis):
        if ei in auto:
            continue
        order = np.argsort(-S[ii])
        for jj in order[:3]:
            aj = ajs[jj]
            s = float(S[ii, jj])
            if s >= REVIEW_MIN and aj not in ar_taken:
                review[ei].append((aj, round(s, 1)))

print(f"new auto matches: {len(auto)} | review rows: {len(review)}")
scores = sorted([s for _, s in auto.values()], reverse=True)
print("auto score distribution:", scores[:15], "..." if len(scores) > 15 else "")

# spot check
print("\nsample auto matches:")
for ei, (aj, s) in list(auto.items())[:20]:
    print(f"  {s:5.1f}  {unm_en[ei]['en']}  <->  {unm_ar[aj]['ar']}")
print("\nsample review rows:")
for ei, cands in list(review.items())[:8]:
    u = unm_en[ei]
    print(f"  {u['en']} ({u['d']}/{u['m']}): " +
          "; ".join(f"{unm_ar[aj]['ar']} [{s}]" for aj, s in cands))

# ---------------- reviewer-approved matches -------------------------------
accepted = {}          # en_idx -> (ar_idx, source)
accept_log = []        # (english, arabic, district, status)
for en_n, ar_n, d_n in ACCEPT:
    ei = next((i for i, u in enumerate(unm_en)
               if u["en"] == en_n and i not in auto), None)
    aj = next((j for j, u in enumerate(unm_ar)
               if u["ar"] == ar_n and j not in ar_taken), None)
    if ei is None:
        accept_log.append((en_n, ar_n, d_n, "skipped — english not unmatched"))
        continue
    if aj is None:
        accept_log.append((en_n, ar_n, d_n, "skipped — arabic not unmatched"))
        continue
    eu, au = unm_en[ei], unm_ar[aj]
    # compound arabic district labels count as every district they name
    ok = (eu["dkey"] and eu["mkey"] and au["mkey"] and
          (eu["dkey"], eu["mkey"]) in {(dk, au["mkey"]) for dk in dkeys_ar(au["d"])})
    if norm_lat(d_n) != norm_lat(eu["d"] or "") or not ok:
        accept_log.append((en_n, ar_n, d_n, "skipped — district/mohafaza mismatch"))
        continue
    accepted[ei] = (aj, "Reviewer-approved")
    ar_taken.add(aj)
    review.pop(ei, None)   # an approved english leaves the review pile
    accept_log.append((en_n, ar_n, d_n, "applied"))
if ACCEPT:
    n_ok = sum(1 for r in accept_log if r[3] == "applied")
    print(f"reviewer-approved matches: {n_ok} applied, {len(accept_log) - n_ok} skipped")

# ---------------- subdivision discovery ------------------------------------
# A subdivision = a still-unmatched arabic row belonging to a definite match
# in the same district+mohafaza. Three ways to recognize one:
#  (i)  qualifier-stripped bases agree ('الفرزل الفوقا' -> 'الفرزل'), with the
#       fixed SUBDIV_QUALS list; bases must be >= 4 chars;
#  (ii) the arabic row is the seed's full arabic name plus extra words
#       ('القبيات الذوق' belongs to seed 'القبيات') — generalizes (i) beyond
#       the fixed qualifier list; seed name must be >= 4 chars;
#  (iii) short-base حي quarters: 'صور حي الجامع' belongs to seed 'صور'.
#       'حي' (quarter) is unambiguous, so no base-length floor is needed.
# The district+mohafaza check is what keeps cross-district lookalikes apart
# (e.g. 'الجديدة حي ...' rows never attach to a 'الجديدة' matched elsewhere).
# Pass A (seeded): definite matches (v5 + auto + reviewer-approved) attract
# their subdivisions.
# Pass B (unseeded): an unmatched english matching the shared base of 2+
# unmatched arabic rows in its district takes the whole group.
subdiv = defaultdict(list)   # unm_en idx -> [(ar_idx, score)]
already_subdiv = []          # (en_name, ar_idx, d, m, source) for already-seeds
discovered = []              # (en, ar, d, m, source) for the Summary sheet

seeds = []  # (en_name, ar_name, dkey, mkey, d_raw, m_raw, en_idx_or_None)
for rec in already:
    dk, mk = dkey_en(rec["d"]), mkey_en(rec["m"])
    if dk and mk:
        seeds.append((rec["en"], rec["ar"], dk, mk, rec["d"], rec["m"], None))
for ei, (aj, _s) in auto.items():
    u = unm_en[ei]
    seeds.append((u["en"], unm_ar[aj]["ar"], u["dkey"], u["mkey"], u["d"], u["m"], ei))
for ei, (aj, _src) in accepted.items():
    # reviewer-approved matches seed the subdivision pass like auto matches
    u = unm_en[ei]
    seeds.append((u["en"], unm_ar[aj]["ar"], u["dkey"], u["mkey"], u["d"], u["m"], ei))

for en_name, ar_name, dk, mk, d_raw, m_raw, ei in seeds:
    base = subdiv_base(ar_name)
    # exact seed name (parens stripped), for rules (ii) and (iii)
    seed_full = re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)\s*", " ", norm_ar(ar_name or ""))).strip()
    seed_long = len(seed_full.replace(" ", "")) >= 4
    if not base and not seed_long and not seed_full:
        continue
    for j in groups_ar.get((dk, mk), []):
        if j in ar_taken:
            continue
        ajr = unm_ar[j]
        arj = norm_ar(ajr["ar"])
        b2 = subdiv_base(ajr["ar"])
        hit = False
        if base and b2 and b2 == base:
            hit = True                                        # (i) qualifier-stripped base
        elif seed_long and arj.startswith(seed_full + " "):
            hit = True                                        # (ii) seed name + extra words
        elif seed_full and "حي" in arj.split():
            toks = arj.split()
            if " ".join(toks[:toks.index("حي")]) == seed_full:
                hit = True                                    # (iii) short-base حي quarter
        if hit:
            s = pair_score(en_name_variants(en_name, dk), ajr["ar"])
            src = f"Fuzzy-auto subdivision ({s:.1f})"
            if ei is None:
                already_subdiv.append((en_name, j, d_raw, m_raw, src))
            else:
                subdiv[ei].append((j, s))
            ar_taken.add(j)
            discovered.append((en_name, ajr["ar"], d_raw, m_raw, src + " [seeded]"))

for key in sorted(set(groups_en) & set(groups_ar)):
    for ei in groups_en[key]:
        if ei in auto or ei in accepted or ei in subdiv:
            continue
        u = unm_en[ei]
        by_base = defaultdict(list)
        for j in groups_ar[key]:
            if j in ar_taken:
                continue
            b = subdiv_base(unm_ar[j]["ar"])
            if b:
                by_base[b].append(j)
        for b, js in sorted(by_base.items()):
            u_variants = en_name_variants(u["en"], u["dkey"])
            if len(js) >= 2 and pair_score(u_variants, b) >= AUTO_MIN:
                for j in js:
                    s = pair_score(u_variants, unm_ar[j]["ar"])
                    subdiv[ei].append((j, s))
                    ar_taken.add(j)
                    discovered.append((u["en"], unm_ar[j]["ar"], u["d"], u["m"],
                                       f"Fuzzy-auto subdivision ({s:.1f}) [unseeded]"))

print(f"subdivision pairs discovered: {len(discovered)} "
      f"({len(subdiv_groups_v5)} groups already in v5 source)")
for en_name, ar_name, _d, _m, src in discovered[:15]:
    print(f"  subdiv  {en_name}  <->  {ar_name}  ({src})")

# drop taken arabic rows from review lists
for ei in list(review):
    review[ei] = [(aj, s) for aj, s in review[ei] if aj not in ar_taken]
    if not review[ei]:
        del review[ei]

# ---------------- community suggestions (from the website) ----------------
SUGGEST_CSV = REPO / "FuzzyMatch" / "suggested_matches.csv"
suggested = {}        # en_idx -> (ar_idx, suggester)
suggest_log = []      # (english, arabic, suggester, status)
suggest_stats = defaultdict(int)

if SUGGEST_CSV.exists():
    import csv
    with open(SUGGEST_CSV, encoding="utf-8-sig") as f:
        srows = list(csv.DictReader(f))
    srows.sort(key=lambda r: r.get("timestamp_utc") or "")
    en_by_exact = {u["en"]: i for i, u in enumerate(unm_en)}
    en_by_norm = {norm_lat(u["en"]): i for i, u in enumerate(unm_en)}
    ar_by_exact = {u["ar"]: j for j, u in enumerate(unm_ar)}
    ar_by_norm = {norm_ar(u["ar"]): j for j, u in enumerate(unm_ar)}

    def find_en(name):
        if name in en_by_exact:
            return en_by_exact[name]
        return en_by_norm.get(norm_lat(name))

    def find_ar(name):
        if name in ar_by_exact:
            return ar_by_exact[name]
        return ar_by_norm.get(norm_ar(name))

    for r in srows:
        en_n = (r.get("english_name") or "").strip()
        ar_n = (r.get("arabic_name") or "").strip()
        suggester = (r.get("suggester") or "").strip()
        if not en_n or not ar_n:
            suggest_stats["skipped_empty"] += 1
            suggest_log.append((en_n, ar_n, suggester, "skipped — empty name"))
            continue
        ei, aj = find_en(en_n), find_ar(ar_n)
        if ei is None:
            suggest_stats["skipped_en_gone"] += 1
            suggest_log.append((en_n, ar_n, suggester, "skipped — english not unmatched"))
            continue
        if aj is None:
            suggest_stats["skipped_ar_gone"] += 1
            suggest_log.append((en_n, ar_n, suggester, "skipped — arabic not unmatched"))
            continue
        if ei in suggested and suggested[ei][0] == aj:
            suggest_stats["duplicate"] += 1
            suggest_log.append((en_n, ar_n, suggester, "duplicate — already applied"))
            continue
        if ei in auto or ei in suggested:
            suggest_stats["conflict_en"] += 1
            suggest_log.append((en_n, ar_n, suggester, "conflict — english already matched"))
            continue
        if aj in ar_taken:
            suggest_stats["conflict_ar"] += 1
            suggest_log.append((en_n, ar_n, suggester, "conflict — arabic already taken"))
            continue
        eu, au = unm_en[ei], unm_ar[aj]
        if (eu["dkey"] and eu["mkey"] and au["dkey"] and au["mkey"]
                and (eu["dkey"], eu["mkey"]) != (au["dkey"], au["mkey"])):
            suggest_stats["skipped_district"] += 1
            suggest_log.append((en_n, ar_n, suggester, "skipped — district/mohafaza mismatch"))
            continue
        suggested[ei] = (aj, suggester)
        ar_taken.add(aj)
        review.pop(ei, None)   # a suggested english leaves the review pile
        suggest_stats["applied"] += 1
        suggest_log.append((en_n, ar_n, suggester, "applied"))
    n_bad = sum(v for k, v in suggest_stats.items() if k != "applied")
    print(f"community suggestions: {suggest_stats['applied']} applied, {n_bad} skipped/conflicted")
else:
    print("community suggestions: no suggested_matches.csv found")

# ---------------- write workbook ----------------
wb = openpyxl.Workbook()
ws = wb.active
# english indices claimed by any automatic match are decided: they must not
# appear in Review, Unmatched, or "still unmatched" counts
decided_en = set(auto) | set(suggested) | set(subdiv) | set(accepted)

ws.title = "Summary"
ws.append(["Lebanese Villages - Fuzzy Arabic<->English Matching v6 (rerun)"])
ws.append(["Date", date.today().isoformat()])
ws.append(["Source file", str(V5)])
ws.append(["Method", "Offline Arabic->Latin transliteration (ya y/i, ta t/a/e variants) + "
           "French-dialect EN normalization; english names also tried with a leading "
           "district-name token stripped ('Tripoli Al-Tabbaneh' -> 'Al-Tabbaneh'); "
           "rapidfuzz WRatio; Hungarian one-to-one "
           "per (district, mohafaza) group; subdivision pass attaches qualifier-sharing "
           "arabic rows (جنوبي/شمالي/شرقي/غربي/فوقا/تحتا/حي …), 'seed + extra words' rows, "
           "and short-base حي quarters to a definite match; french 'X et Y' compounds "
           "scored per-side against arabic 'X و Y'"]) 
ws.append(["Constraint", "Candidates ONLY from the same district AND mohafaza as the English village"])
ws.append(["Community suggestions", "auto-accepted from FuzzyMatch/suggested_matches.csv "
           "(earliest timestamp wins on conflicts; district+mohafaza enforced)"])
ws.append(["Auto-match threshold", f">= {AUTO_MIN}"])
ws.append(["Review band", f"{REVIEW_MIN}-{AUTO_MIN} (top-3 candidates, human confirm)"])
ws.append([None])
ws.append(["Original matched pairs (v5, subdivisions expanded)", len(already)])
ws.append(["Unmatched English before rerun", len(unm_en)])
ws.append(["Unmatched Arabic before rerun", len(unm_ar)])
ws.append(["New auto matches", len(auto)])
ws.append(["Review candidate rows", sum(1 for ei in review if ei not in decided_en)])
ws.append(["Community-suggested matches applied", len(suggested)])
ws.append(["Reviewer-approved matches applied (--accept)", len(accepted)])
ws.append(["Suggestions skipped/conflicted/duplicated",
           sum(v for k, v in suggest_stats.items() if k != "applied")])
still_en = len(unm_en) - len(auto) - sum(1 for ei in review if ei not in auto)
ws.append(["Still unmatched English", len(unm_en) - len(decided_en)])
ws.append(["Still unmatched Arabic", len(unm_ar) - len(ar_taken)])
ws.append(["Arabic rows dropped as already-matched (one-to-one)", len(dup_arabic)])
ws.append(["One-to-one conflicts needing human review", len(conflicts)])
ws.append(["Subdivision groups in source (expanded to pairs)", len(subdiv_groups_v5)])
ws.append(["Subdivision pairs auto-discovered", len(discovered)])
if subdiv_groups_v5:
    ws.append([None])
    ws.append(["SUBDIVISION GROUPS (source workbook)"])
    ws.append(["English Name", "District", "Mohafaza", "Arabic subdivisions"])
    for en, d, m, parts in sorted(subdiv_groups_v5, key=lambda x: str(x[0])):
        ws.append([en, d, m, " | ".join(parts)])
if discovered:
    ws.append([None])
    ws.append(["SUBDIVISION PAIRS AUTO-DISCOVERED"])
    ws.append(["English Name", "Arabic Name", "District", "Mohafaza", "Source"])
    for row in discovered:
        ws.append(list(row))
if accept_log:
    ws.append([None])
    ws.append(["REVIEWER-APPROVED MATCHES (--accept overrides)"])
    ws.append(["English Name", "Arabic Name", "District", "Status"])
    for row in accept_log:
        ws.append(list(row))
if dup_arabic:
    ws.append([None])
    ws.append(["DROPPED — arabic village already matched in the same district+mohafaza"])
    ws.append(["Arabic Name", "District", "Mohafaza", "Matched English"])
    for ar, d, m, ens in sorted(dup_arabic, key=lambda x: norm_ar(x[0])):
        ws.append([ar, d, m, " | ".join(ens)])
if conflicts:
    ws.append([None])
    ws.append(["CONFLICTS — one arabic+district claimed by 2+ english names (resolve in v5 source)"])
    ws.append(["Arabic (normalized)", "District", "Mohafaza", "English names"])
    for (ar, dk, mk), ens in sorted(conflicts.items()):
        ws.append([ar, dk, mk, " | ".join(ens)])

we = wb.create_sheet("English (Full)")
we.append(["English Name", "Arabic Name", "District Name", "Mohafaza", "Match Source"])
for rec in already:
    we.append([rec["en"], rec["ar"], rec["d"], rec["m"], rec["src"]])
for en_name, j, d_raw, m_raw, src in already_subdiv:
    we.append([en_name, unm_ar[j]["ar"], d_raw, m_raw, src])
for i, u in enumerate(unm_en):
    if i in auto:
        aj, s = auto[i]
        we.append([u["en"], unm_ar[aj]["ar"], u["d"], u["m"], f"Fuzzy-auto ({s:.1f})"])
    elif i in suggested:
        aj, suggester = suggested[i]
        by = f" (by {suggester})" if suggester else ""
        we.append([u["en"], unm_ar[aj]["ar"], u["d"], u["m"], f"Community suggestion{by}"])
    elif i in accepted:
        aj, src = accepted[i]
        we.append([u["en"], unm_ar[aj]["ar"], u["d"], u["m"], src])
    elif i not in subdiv:
        we.append([u["en"], None, u["d"], u["m"], "Missing"])
    for aj2, s2 in subdiv.get(i, []):
        we.append([u["en"], unm_ar[aj2]["ar"], u["d"], u["m"],
                   f"Fuzzy-auto subdivision ({s2:.1f})"])

wa = wb.create_sheet("Arabic (Full)")
wa.append(["Village Name", "English Name", "District Name", "Mohafaza"])
for rec in ar_full:
    wa.append([rec["ar"], rec["en"], rec["d"], rec["m"]])
# fill English names for auto matches
for ei, (aj, s) in auto.items():
    # find row index in ar_full for this unmatched arabic record
    pass
# (Arabic (Full) keeps original EN column from v5; patch auto/suggested rows)
auto_ar_to_en = {}
for ei, (aj, s) in auto.items():
    auto_ar_to_en[id(unm_ar[aj])] = (unm_en[ei]["en"], s)
for ei, (aj, _suggester) in suggested.items():
    auto_ar_to_en[id(unm_ar[aj])] = (unm_en[ei]["en"], None)
for ei, (aj, _src) in accepted.items():
    auto_ar_to_en[id(unm_ar[aj])] = (unm_en[ei]["en"], None)
for ei, lst in subdiv.items():
    for aj, s in lst:
        auto_ar_to_en[id(unm_ar[aj])] = (unm_en[ei]["en"], s)
for en_name, j, _d, _m, _s in already_subdiv:
    auto_ar_to_en[id(unm_ar[j])] = (en_name, None)
row_idx = 2
for rec in ar_full:
    if id(rec) in auto_ar_to_en and not (rec["en"] and str(rec["en"]).strip()):
        en_name, s = auto_ar_to_en[id(rec)]
        wa.cell(row=row_idx, column=2, value=en_name)
    row_idx += 1

wr = wb.create_sheet("Review")
wr.append(["English Name", "District", "Mohafaza",
           "Candidate 1 (score)", "Candidate 2 (score)", "Candidate 3 (score)"])
for ei in sorted(review, key=lambda i: unm_en[i]["en"]):
    if ei in decided_en:
        continue  # matched via auto/suggested/subdiv — not reviewable
    u = unm_en[ei]
    cells = [u["en"], u["d"], u["m"]]
    for aj, s in review[ei][:3]:
        a = unm_ar[aj]
        hint = f" ({a['d']})" if a["d"] else ""
        cells.append(f"{a['ar']}{hint} [{s}]")
    while len(cells) < 6:
        cells.append(None)
    wr.append(cells)

wu = wb.create_sheet("Unmatched")
wu.append(["Side", "Name", "District", "Mohafaza"])
for i, u in enumerate(unm_en):
    if i not in decided_en:
        wu.append(["English", u["en"], u["d"], u["m"]])
for j, u in enumerate(unm_ar):
    if j not in ar_taken:
        wu.append(["Arabic", u["ar"], u["d"], u["m"]])

wsug = wb.create_sheet("Suggestions")
wsug.append(["English Name", "Arabic Name", "Suggester", "Status"])
for row in suggest_log:
    wsug.append(list(row))

for name in ["DistrictTranslation", "Mohafaza Translations"]:
    src = wb5[name]
    dst = wb.create_sheet(name)
    for r in src.iter_rows(values_only=True):
        dst.append(list(r))

wb.save(OUT)
print("\nwrote", OUT)
wb5.close()
