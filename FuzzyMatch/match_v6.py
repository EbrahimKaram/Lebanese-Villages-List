#!/usr/bin/env python3
"""match_v6.py — rerun of the fuzzy match with a hard constraint:

Review/auto candidates may ONLY come from the same district AND mohafaza
as the English village. Regenerates FuzzyMatch/Lebanese_Villages_Matched_v6.xlsx
from the v5 workbook (v5 itself is untouched).
"""
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import openpyxl
from rapidfuzz import fuzz

REPO = Path(__file__).resolve().parent.parent
V5 = REPO / "FuzzyMatch" / "Grok 4.6" / "Lebanese_Villages_Improved_Matching_v5.xlsx"
OUT = REPO / "FuzzyMatch" / "Lebanese_Villages_Matched_v6.xlsx"

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
    """All transliteration variants -> token lists, article stripped, prefixes dropped."""
    variants = set()
    for ya in ("y", "i"):
        for ta in ("t", "a"):
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


def pair_score(en: str, ar: str) -> float:
    best = 0.0
    for ev in norm_en_tokens(en):
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
for latin, arabic, _m in dist_rows:
    if latin and arabic:
        lat2ar_d[norm_lat(latin)] = arabic
        ar2lat_d[norm_ar(arabic)] = latin
KNOWN_AR_D = list(ar2lat_d.keys())

moh_rows = list(wb5["Mohafaza Translations"].iter_rows(values_only=True))[1:]
ar2lat_m, lat2ar_m = {}, {}
for arabic, latin in moh_rows:
    if arabic and latin:
        ar2lat_m[norm_ar(arabic)] = latin
        lat2ar_m[norm_lat(latin)] = arabic


def dkey_en(latin):
    k = norm_lat(latin)
    return latin if k in lat2ar_d else None


def dkey_ar(arabic):
    c = clean_ar_district(arabic)
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

# already matched pairs (kept as-is)
already = []          # dicts for English (Full)
unm_en, unm_ar = [], []

for r in en_rows:
    en, ar, d, m, src = r[0], r[1], r[2], r[3], r[4]
    if not en or not str(en).strip() or is_placeholder(en):
        continue
    rec = {"en": en, "ar": ar, "d": d, "m": m, "src": src}
    if ar and str(ar).strip():
        already.append(rec)
    else:
        rec["dkey"] = dkey_en(d)
        rec["mkey"] = mkey_en(m)
        unm_en.append(rec)

ar_full = []
for r in ar_rows:
    ar, en, d, m = r[0], r[1], r[2], r[3]
    if not ar or not str(ar).strip():
        continue
    rec = {"ar": ar, "en": en, "d": d, "m": m}
    ar_full.append(rec)
    if not (en and str(en).strip()):
        rec["dkey"] = dkey_ar(d)
        rec["mkey"] = mkey_ar(m)
        unm_ar.append(rec)

print(f"already matched: {len(already)} | unmatched EN: {len(unm_en)} | unmatched AR: {len(unm_ar)}")
no_key_en = sum(1 for u in unm_en if not u["dkey"] or not u["mkey"])
no_key_ar = sum(1 for u in unm_ar if not u["dkey"] or not u["mkey"])
print(f"rows without district/mohafaza key (no candidates possible): EN={no_key_en} AR={no_key_ar}")

# ---------------- group by (dkey, mkey) ----------------
groups_en, groups_ar = defaultdict(list), defaultdict(list)
for i, u in enumerate(unm_en):
    if u["dkey"] and u["mkey"]:
        groups_en[(u["dkey"], u["mkey"])].append(i)
for j, u in enumerate(unm_ar):
    if u["dkey"] and u["mkey"]:
        groups_ar[(u["dkey"], u["mkey"])].append(j)

# ---------------- score + hungarian per group ----------------
auto = {}            # en_idx -> (ar_idx, score)
ar_taken = set()
review = defaultdict(list)   # en_idx -> [(ar_idx, score)]

import numpy as np  # noqa

for key in sorted(set(groups_en) & set(groups_ar)):
    eis, ajs = groups_en[key], groups_ar[key]
    n, m_ = len(eis), len(ajs)
    S = np.zeros((n, m_))
    for ii, ei in enumerate(eis):
        for jj, aj in enumerate(ajs):
            S[ii, jj] = pair_score(unm_en[ei]["en"], unm_ar[aj]["ar"])
    # mutual-best auto matching: each side must be the other's top choice
    en_best = np.argmax(S, axis=1)   # for each en row -> best ar col
    ar_best = np.argmax(S, axis=0)   # for each ar col -> best en row
    for ii, ei in enumerate(eis):
        jj = int(en_best[ii])
        s = float(S[ii, jj])
        if s >= AUTO_MIN and int(ar_best[jj]) == ii:
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

# ---------------- write workbook ----------------
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Summary"
ws.append(["Lebanese Villages - Fuzzy Arabic<->English Matching v6 (rerun)"])
ws.append(["Date", "2026-09-14"])
ws.append(["Source file", "FuzzyMatch/Grok 4.6/Lebanese_Villages_Improved_Matching_v5.xlsx"])
ws.append(["Method", "Offline Arabic->Latin transliteration (ya y/i, ta t/a variants) + "
           "French-dialect EN normalization; rapidfuzz WRatio; Hungarian one-to-one "
           "per (district, mohafaza) group"])
ws.append(["Constraint", "Candidates ONLY from the same district AND mohafaza as the English village"])
ws.append(["Auto-match threshold", f">= {AUTO_MIN}"])
ws.append(["Review band", f"{REVIEW_MIN}-{AUTO_MIN} (top-3 candidates, human confirm)"])
ws.append([None])
ws.append(["Original matched pairs (v5, untouched)", len(already)])
ws.append(["Unmatched English before rerun", len(unm_en)])
ws.append(["Unmatched Arabic before rerun", len(unm_ar)])
ws.append(["New auto matches", len(auto)])
ws.append(["Review candidate rows", len(review)])
still_en = len(unm_en) - len(auto) - sum(1 for ei in review if ei not in auto)
ws.append(["Still unmatched English", len(unm_en) - len(auto)])
ws.append(["Still unmatched Arabic", len(unm_ar) - len(ar_taken)])

we = wb.create_sheet("English (Full)")
we.append(["English Name", "Arabic Name", "District Name", "Mohafaza", "Match Source"])
for rec in already:
    we.append([rec["en"], rec["ar"], rec["d"], rec["m"], rec["src"]])
for i, u in enumerate(unm_en):
    if i in auto:
        aj, s = auto[i]
        we.append([u["en"], unm_ar[aj]["ar"], u["d"], u["m"], f"Fuzzy-auto ({s:.1f})"])
    else:
        we.append([u["en"], None, u["d"], u["m"], "Missing"])

wa = wb.create_sheet("Arabic (Full)")
wa.append(["Village Name", "English Name", "District Name", "Mohafaza"])
for rec in ar_full:
    wa.append([rec["ar"], rec["en"], rec["d"], rec["m"]])
# fill English names for auto matches
for ei, (aj, s) in auto.items():
    # find row index in ar_full for this unmatched arabic record
    pass
# (Arabic (Full) keeps original EN column from v5; patch auto-matched rows)
auto_ar_to_en = {}
for ei, (aj, s) in auto.items():
    auto_ar_to_en[id(unm_ar[aj])] = (unm_en[ei]["en"], s)
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
    if i not in auto:
        wu.append(["English", u["en"], u["d"], u["m"]])
for j, u in enumerate(unm_ar):
    if j not in ar_taken:
        wu.append(["Arabic", u["ar"], u["d"], u["m"]])

for name in ["DistrictTranslation", "Mohafaza Translations"]:
    src = wb5[name]
    dst = wb.create_sheet(name)
    for r in src.iter_rows(values_only=True):
        dst.append(list(r))

wb.save(OUT)
print("\nwrote", OUT)
wb5.close()
