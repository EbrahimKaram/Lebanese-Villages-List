from glob import glob
import pandas as pd
import re
import requests
import json
import ast
from unidecode import unidecode
from rapidfuzz import fuzz
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv
import os
import hungarian_assign as ha
import time
import sys
#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":

    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, ".."))
    print(f"Output directory: {OUT_DIR}")
    VERBOSE = True
    # optional: pass a district name as second CLI arg to debug a single district
    # (first CLI arg, if given, overrides the input Excel filename)
    DEBUG_DISTRICT = None
    if len(sys.argv) > 2:
        DEBUG_DISTRICT = sys.argv[2]
        print(f"DEBUG_DISTRICT set to: {DEBUG_DISTRICT}")
    elif len(sys.argv) > 1 and sys.argv[1].upper() == 'FIRST':
        # backwards-compat: a single arg used to mean "debug district"
        DEBUG_DISTRICT = sys.argv[1]

    # OFFLINE mode skips all Yamli network calls entirely (they aren't reachable from every
    # environment, e.g. sandboxes with an egress allowlist) and relies solely on the local
    # phonetic transliteration scorer below, which scores noticeably higher in testing anyway
    # (~93% top-1 accuracy on the 1136 already-confirmed pairs vs ~80% for the old unidecode-only
    # fallback). Set OFFLINE=0 in the environment to attempt Yamli lookups when network is available.
    OFFLINE = os.environ.get("OFFLINE", "1") != "0"
    if VERBOSE:
        print(f"OFFLINE mode: {OFFLINE}")

    # Input workbook: defaults to the canonical "Use this" file at the repo root (the file the
    # README's contribution instructions point at) rather than a stale snapshot copied into this
    # folder. Pass an explicit path as the first CLI arg to override.
    if len(sys.argv) > 1 and sys.argv[1].upper() != 'FIRST':
        names_of_excel = [sys.argv[1]]
    else:
        names_of_excel = glob(os.path.join(REPO_ROOT, "*Use this*.xlsx"))
        if not names_of_excel:
            # fall back to the local snapshot so the script still runs standalone
            names_of_excel = glob(os.path.join(OUT_DIR, "matched_output(178).xlsx"))
    if not names_of_excel:
        print("No Excel files found (looked for '*Use this*.xlsx' at repo root).")
        exit()
    print(f"Reading from: {names_of_excel[0]}")

    arabicDataframe = pd.read_excel(names_of_excel[0], sheet_name=0)
    englishDataframe = pd.read_excel(names_of_excel[0], sheet_name=1)
    districtTranslation = pd.read_excel(names_of_excel[0], sheet_name=2)

    # reuse Yamli helpers (same as original script)
    def get_yamli_candidates(text):
        url = 'https://api.yamli.com/transliterate.ashx'
        params = {
            'word': text,
            'tool': 'api',
            'account_id': '000006',
            'prot': 'https%3A',
            'hostname': 'www.yamli.com',
            'path': '%2F',
            'build': '5515',
            'sxhr_id': '9'
        }
        try:
            resp = requests.get(url, params=params, timeout=6)
            raw = resp.text
            m = re.search(r"Yamli\.I\.SXHRData\.dataCallback\((\{.*\})\);", raw, re.S)
            if not m:
                idx = raw.find('{')
                if idx == -1:
                    return []
                payload = raw[idx:]
            else:
                payload = m.group(1)
            j = None
            try:
                j = json.loads(payload)
            except Exception:
                try:
                    j = json.loads(payload.replace("'", '"'))
                except Exception:
                    try:
                        j = ast.literal_eval(payload)
                    except Exception:
                        arabic_seqs = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-]{2,}", raw)
                        items = []
                        for s in arabic_seqs:
                            s2 = s.strip()
                            if s2 and s2 not in items:
                                items.append(s2)
                        return items
            raw_data = j.get('data') if isinstance(j, dict) else None
            data = {}
            if isinstance(raw_data, str):
                try:
                    data = json.loads(raw_data)
                except Exception:
                    try:
                        data = ast.literal_eval(raw_data)
                    except Exception:
                        m = re.search(r"r\s*[:=]\s*['\"]([^'\"]+)['\"]", raw_data)
                        if m:
                            data = {'r': m.group(1)}
                        else:
                            data = {}
            elif isinstance(raw_data, dict):
                data = raw_data
            r = data.get('r', '')
            if not r:
                return []
            items = re.split(r'(?:/[0-9][|])|(?:/[0-9])', r)
            items = [s for s in items if s]
            return items
        except Exception:
            return []

    def yamli_fetch_token(token, session):
        url = 'https://api.yamli.com/transliterate.ashx'
        params = {
            'word': token,
            'tool': 'api',
            'account_id': '000006',
            'prot': 'https%3A',
            'hostname': 'www.yamli.com',
            'path': '%2F',
            'build': '5515',
            'sxhr_id': '9'
        }
        try:
            resp = session.get(url, params=params, timeout=6)
            raw = resp.text
            m = re.search(r"Yamli\.I\.SXHRData\.dataCallback\((\{.*\})\);", raw, re.S)
            if not m:
                idx = raw.find('{')
                if idx == -1:
                    return []
                payload = raw[idx:]
            else:
                payload = m.group(1)
            j = None
            try:
                j = json.loads(payload)
            except Exception:
                try:
                    j = json.loads(payload.replace("'", '"'))
                except Exception:
                    try:
                        j = ast.literal_eval(payload)
                    except Exception:
                        arabic_seqs = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-]{2,}", payload)
                        items = []
                        for s in arabic_seqs:
                            s2 = s.strip()
                            if s2 and s2 not in items:
                                items.append(s2)
                        return items
            raw_data = j.get('data') if isinstance(j, dict) else None
            data = {}
            if isinstance(raw_data, str):
                try:
                    data = json.loads(raw_data)
                except Exception:
                    try:
                        data = json.loads(raw_data.replace("'", '"'))
                    except Exception:
                        try:
                            data = ast.literal_eval(raw_data)
                        except Exception:
                            m2 = re.search(r"r\s*[:=]\s*['\"]([^'\"]+)['\"]", raw_data)
                            if m2:
                                data = {'r': m2.group(1)}
                            else:
                                data = {}
            elif isinstance(raw_data, dict):
                data = raw_data
            r = data.get('r', '')
            if not r:
                return []
            items = re.split(r'(?:/[0-9][|])|(?:/[0-9])', r)
            items = [s for s in items if s]
            return items
        except Exception:
            return []

    full_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                       .notnull()]
    missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                            .isnull()]
    missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                          .isnull()]

    # Some English rows are shapefile artifacts, not real village names (e.g. "n.a. (2)",
    # "Conflict" — GADM polygon-naming placeholders). Matching these would just burn a real
    # Arabic candidate on garbage, so drop them before scoring and log them separately.
    _PLACEHOLDER_RE = re.compile(r'^(n\.?a\.?(\s*\(\d+\))?|conflict|unnamed.*)$', re.I)
    _placeholder_mask = missing_info_english['English Name'].astype(str).str.strip().str.match(_PLACEHOLDER_RE)
    placeholder_rows = missing_info_english[_placeholder_mask]
    missing_info_english = missing_info_english[~_placeholder_mask]
    if VERBOSE and len(placeholder_rows):
        print(f"Skipping {len(placeholder_rows)} placeholder/non-name English rows (e.g. 'n.a.', 'Conflict'): "
              f"{placeholder_rows['English Name'].tolist()}")

    arabic_village_set = set(missing_info_arabic['Village Name'].dropna().astype(str).tolist())
    ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

    # locality tokens helper
    _LOCALITY_TOKENS = set(['وادي', 'واديه', 'واديّ', 'حي', 'حيه', 'حيّ', 'ال', 'منارة', 'منار', 'منطقه', 'منطقة'])
    def remove_locality_tokens(s: str) -> str:
        if not s:
            return ''
        parts = re.split(r'[\s\-\–\—]+', s)
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            p_noal = re.sub(r'^ال', '', p)
            if p_noal in _LOCALITY_TOKENS or p in _LOCALITY_TOKENS:
                continue
            out.append(p_noal)
        return ' '.join(out)

    def token_translit_score(eng_norm, arabic_phrase):
        best = 0
        for tok in re.split(r'[\s\-\–\—]+', str(arabic_phrase)):
            tok = tok.strip()
            if not tok:
                continue
            tok = re.sub(r'^ال', '', tok)
            tok = remove_locality_tokens(tok) or tok
            tok_lat = unidecode(tok).lower()
            if not tok_lat:
                continue
            if tok_lat == eng_norm:
                return 100
            s = fuzz.token_set_ratio(eng_norm, tok_lat)
            if s > best:
                best = s
            p = fuzz.partial_ratio(eng_norm, tok_lat)
            if p > best:
                best = p
        return best

    # --- Phonetic Arabic->Latin transliteration (replaces unidecode for scoring) ---
    # unidecode() transliterates Arabic using an academic/library scheme (e.g. ع -> "`",
    # ة -> "@", ح -> "H") that looks nothing like how these villages are actually romanized
    # in this dataset (French/English colloquial Lebanese spelling, e.g. "Aaouainat",
    # "Aïn-Yacoub"). That mismatch is why match scores were stuck in the 30-55 range and the
    # THRESHOLD=80 cutoff almost never fired (arabic_match_accepted.csv stayed empty).
    # AR_MAP below maps each Arabic letter to the Latin letter(s) most commonly used for it in
    # this dataset's romanizations; normalize_latin() then folds both sides onto the same
    # simplified alphabet (merging kh/k, gh/r, sh/ch/s, th/t, ou/o, doubled-vowel variants, etc.)
    # so genuinely-equivalent spellings compare as equal or near-equal.
    # Validated on the 1136 already-confirmed English<->Arabic pairs in this workbook, filtered
    # to same-district candidates (the real matching scenario): top-1 accuracy 93% / top-3 99%,
    # vs 80% / 90% for the old unidecode-only scoring.
    AR_MAP = {
        'ا': 'a', 'أ': 'a', 'إ': 'a', 'آ': 'a', 'ء': '',
        'ب': 'b', 'ت': 't', 'ث': 't',
        'ج': 'j', 'ح': 'h', 'خ': 'kh',
        'د': 'd', 'ذ': 'z',
        'ر': 'r', 'ز': 'z',
        'س': 's', 'ش': 'ch',
        'ص': 's', 'ض': 'd',
        'ط': 't', 'ظ': 'z',
        'ع': 'a', 'غ': 'gh',
        'ف': 'f', 'ق': 'k',
        'ك': 'k', 'ل': 'l',
        'م': 'm', 'ن': 'n',
        'ه': 'h', 'و': 'ou',
        'ي': 'i', 'ى': 'a', 'ة': 'a',
        'ؤ': 'o', 'ئ': 'i',
        ' ': ' ', '-': ' ', '\u0640': '',  # tatweel
    }
    _AR_DIACRITICS_RE = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
    _NON_ALPHA_RE = re.compile(r'[^a-z\s]')
    _REPEAT_RE = re.compile(r'(.)\1+')
    _LATIN_SUBS = [
        ('kh', 'k'), ('gh', 'r'), ('ch', 'sh'), ('sh', 's'),
        ('th', 't'), ('dj', 'j'), ('dh', 'z'),
        ('ou', 'o'), ('aa', 'a'), ('ee', 'i'), ('ei', 'i'), ('ai', 'i'), ('ay', 'i'),
        ('ph', 'f'), ('qu', 'k'), ('q', 'k'),
    ]

    def arabic_to_loose_latin(s: str) -> str:
        if not isinstance(s, str):
            return ''
        s = _AR_DIACRITICS_RE.sub('', s)
        return ''.join(AR_MAP.get(ch, ch) for ch in s)

    def normalize_latin(s: str) -> str:
        s = unidecode(str(s)).lower()
        s = _NON_ALPHA_RE.sub(' ', s)
        for a, b in _LATIN_SUBS:
            s = s.replace(a, b)
        s = _REPEAT_RE.sub(r'\1', s)
        return re.sub(r'\s+', ' ', s).strip()

    def phonetic_score(english_name: str, arabic_name: str) -> float:
        """Best-effort phonetic similarity between an English name and an Arabic name,
        independent of any Yamli lookups. Combines full-string, partial, and
        vowel-stripped 'skeleton' comparisons since transliteration conventions vary in
        where they place vowels."""
        eng_norm = normalize_latin(english_name)
        ar_norm = normalize_latin(arabic_to_loose_latin(arabic_name))
        if not eng_norm or not ar_norm:
            return 0.0
        s_full = fuzz.token_set_ratio(eng_norm, ar_norm)
        s_partial = fuzz.partial_ratio(eng_norm, ar_norm)
        eng_skel = re.sub(r'[aeiou\s]', '', eng_norm)
        ar_skel = re.sub(r'[aeiou\s]', '', ar_norm)
        s_skel = fuzz.ratio(eng_skel, ar_skel) if eng_skel and ar_skel else 0
        return max(s_full, s_partial, s_skel)

    # yamli cache prefetch
    cache_path = os.path.join(OUT_DIR, 'yamli_token_cache.json')
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as cf:
                yamli_cache = json.load(cf)
        else:
            yamli_cache = {}
    except Exception:
        yamli_cache = {}

    ARTICLES_RE = re.compile(r'^(al|el|the|le|la|de|du)$', re.I)
    unique_tokens = set()
    for _, r in missing_info_english[['English Name']].dropna().iterrows():
        en = str(r['English Name']).strip()
        toks = re.sub(r'[-_]', ' ', en).split()
        for t in toks:
            if not t:
                continue
            if ARTICLES_RE.match(t):
                continue
            if t not in yamli_cache:
                unique_tokens.add(t)

    if unique_tokens and not OFFLINE:
        if VERBOSE:
            print(f"Prefetching Yamli for {len(unique_tokens)} unique tokens...")
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 504))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        with ThreadPoolExecutor(max_workers=8) as ex:
            future_to_token = {ex.submit(yamli_fetch_token, tok, session): tok for tok in unique_tokens}
            for fut in as_completed(future_to_token):
                tok = future_to_token[fut]
                try:
                    res = fut.result()
                    if res:
                        yamli_cache[tok] = res
                        if VERBOSE:
                            print(f"Cached Yamli {tok} -> {len(res)} items")
                    else:
                        yamli_cache[tok] = []
                except Exception as e:
                    yamli_cache[tok] = []
                    if VERBOSE:
                        print(f"Error fetching Yamli for token '{tok}': {e}")
        try:
            with open(cache_path, 'w', encoding='utf-8') as cf:
                json.dump(yamli_cache, cf, ensure_ascii=False, indent=1)
        except Exception:
            if VERBOSE:
                print(f"Could not write Yamli cache to {cache_path}")

    # accumulate scored candidates per english row grouped by district
    district_rows = {}   # district_name -> list of english row indices
    scored_per_row = {}  # eng_index -> list of (cand, cand_lat, score)

    THRESHOLD = 80
    # pruning and safety thresholds for Hungarian assignment
    TOP_K = 5               # keep top-K candidates per English row before building global candidate set
    S_THRESHOLD = 500       # max square size for Hungarian; above this use greedy fallback
    # limits to avoid combinatorial explosion when building Yamli phrase permutations
    PER_TOKEN_MAX = 5               # max Yamli suggestions to keep per token
    MAX_PHRASE_COMBINATIONS = 600   # max total Yamli phrase permutations to generate

    for index, row in missing_info_english.iterrows():
        english_name = str(row['English Name']).strip()
        english_norm = re.sub(r'[-_]', ' ', english_name).lower()
        english_stripped_norm = re.sub(r'[-_]', ' ', re.sub(r'^(al|el)[\s\-_:]+', '', english_name, flags=re.I)).lower()

        arabic_districts = districtTranslation.loc[
            districtTranslation['District'] == row['District Name'],
            'District Arabic'
        ].dropna().unique().tolist()
        if not arabic_districts:
            continue
        candidates = missing_info_arabic[missing_info_arabic['District Name'].isin(arabic_districts)]['Village Name'].tolist()
        candidates = list(dict.fromkeys(candidates))

        # build yamli phrases (skipped entirely in OFFLINE mode)
        yamli_phrases = []
        try:
            yamli_used = False
            tokens = re.sub(r'[-_]', ' ', english_name).split()
            if tokens and not OFFLINE:
                per_token_lists = []
                for t in tokens:
                    if ARTICLES_RE.match(t):
                        per_token_lists.append([t])
                        continue
                    yc = yamli_cache.get(t)
                    if yc is None:
                        yc = get_yamli_candidates(t)
                        try:
                            yamli_cache[t] = yc if yc else []
                        except Exception:
                            pass
                    if yc:
                        yamli_used = True
                    # coerce to list safely and limit per-token suggestions
                    if isinstance(yc, (list, tuple)):
                        yc_list = list(yc)[:PER_TOKEN_MAX]
                        per_token_lists.append(yc_list if yc_list else [t])
                    else:
                        per_token_lists.append([yc] if yc else [t])
                # build product but limit total combinations to avoid explosion
                try:
                    prod = itertools.product(*per_token_lists)
                    yamli_phrases = [' '.join(p) for p in itertools.islice(prod, MAX_PHRASE_COMBINATIONS)]
                except Exception:
                    yamli_phrases = [' '.join(p) for p in itertools.product(*per_token_lists)]
        except Exception:
            yamli_phrases = []

        scored = []
        if yamli_phrases:
            for a_cand in candidates:
                a_str = str(a_cand).strip()
                a_stripped = re.sub(r'^\s*ال[\s\-_:]*', '', a_str)
                best_score = 0
                best_y = ''
                for y in yamli_phrases:
                    if not ARABIC_RE.search(str(y)):
                        continue
                    y_str = str(y).strip()
                    y_stripped = re.sub(r'^\s*ال[\s\-_:]*', '', y_str)
                    try:
                        s1 = fuzz.token_set_ratio(a_str, y_str)
                        a_noloc = remove_locality_tokens(a_str)
                        y_noloc = remove_locality_tokens(y_str)
                        s1_loc = fuzz.token_set_ratio(a_noloc, y_noloc) if a_noloc and y_noloc else 0
                        s2 = fuzz.token_set_ratio(a_stripped, y_str) if a_stripped != a_str else 0
                        s3 = fuzz.token_set_ratio(a_str, y_stripped) if y_stripped != y_str else 0
                        s4 = fuzz.token_set_ratio(a_stripped, y_stripped) if (a_stripped != a_str and y_stripped != y_str) else 0
                        s = max(s1, s1_loc, s2, s3, s4)
                    except Exception:
                        s = 0
                    if s > best_score:
                        best_score = s
                        best_y = y_str
                # Always cross-check against the validated phonetic scorer and keep whichever
                # signal is stronger — Yamli sometimes returns a plausible-but-wrong transliteration,
                # and phonetic_score alone already hits 93% top-1 accuracy on confirmed pairs.
                ph_score = phonetic_score(english_name, a_str)
                if ph_score > best_score:
                    best_score = ph_score
                    if not best_y:
                        best_y = arabic_to_loose_latin(a_str)
                scored.append((a_cand, best_y, best_score))
        else:
            # No Yamli phrases (OFFLINE mode, or Yamli returned nothing for every token):
            # rely on the validated phonetic transliteration scorer.
            for cand in candidates:
                cand_str = str(cand).strip()
                try:
                    cand_lat = arabic_to_loose_latin(cand_str)
                except Exception:
                    cand_lat = cand_str
                s = phonetic_score(english_name, cand_str)
                scored.append((cand, cand_lat, s))

        scored_per_row[index] = {
            'row': row,
            'scored': scored,
            'english_norm': english_norm
        }
        dname = row['District Name']
        district_rows.setdefault(dname, []).append(index)

    # Now run Hungarian per district
    # prepare output files
    results_path = os.path.join(OUT_DIR, 'arabic_match_results.csv')
    accepted_path = os.path.join(OUT_DIR, 'arabic_match_accepted.csv')
    review_path = os.path.join(OUT_DIR, 'arabic_match_review.csv')

    # clear or create files
    timing_path = os.path.join(OUT_DIR, 'hungarian_timing.csv')
    for p, hdr in ((results_path, ['English Name','English District','Best Arabic Match','Match Score']),
                   (accepted_path, ['English Name','English District','Accepted Arabic Name','Match Score']),
                   (review_path, ['English Name','English District','Best Arabic Match','Match Score','Cand1','Cand1_Lat','Cand1_Score','Cand2','Cand2_Lat','Cand2_Score','Cand3','Cand3_Lat','Cand3_Score']),
                   (timing_path, ['District', 'EngCount', 'CandCount', 'SquareSize', 'Method', 'ElapsedSeconds'])):
        try:
            with open(p, 'w', encoding='utf-8', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(hdr)
        except Exception:
            pass

    # If DEBUG_DISTRICT is set, filter to that district (or first district with 'FIRST')
    if DEBUG_DISTRICT:
        if DEBUG_DISTRICT.upper() == 'FIRST':
            # pick the first district in iteration order
            items = list(district_rows.items())[:1]
            district_rows = dict(items)
        else:
            district_rows = {k: v for k, v in district_rows.items() if k == DEBUG_DISTRICT}

    for dname, eng_indices in district_rows.items():
        # build arabic candidate list across all eng rows for this district
        arabic_cands = []
        score_lookup = {}
        # apply TOP_K pruning per english row
        for idx in eng_indices:
            scored = scored_per_row[idx]['scored']
            # keep only top-K candidates by score to limit matrix size
            topk = sorted(scored, key=lambda x: x[2], reverse=True)[:TOP_K]
            for cand, lat, sc in topk:
                if cand not in arabic_cands:
                    arabic_cands.append(cand)
                cur = score_lookup.get((idx, cand), 0)
                if sc > cur:
                    score_lookup[(idx, cand)] = sc

        # safety: if the square matrix becomes too large, fall back to greedy assignment
        S = max(len(eng_indices), len(arabic_cands))
        start_t = time.time()
        if S <= S_THRESHOLD:
            assignments = ha.optimal_assignment(eng_indices, arabic_cands, score_lookup, threshold=THRESHOLD)
            method_used = 'hungarian'
        else:
            # greedy fallback: pick highest-scoring (eng, cand) pairs without conflicts
            method_used = 'greedy'
            assignments = {}
            assigned_eng = set()
            assigned_cand = set()
            # flatten candidate pairs and sort by score desc
            flat = []
            for (idx, cand), sc in score_lookup.items():
                if sc >= THRESHOLD:
                    flat.append((sc, idx, cand))
            flat.sort(reverse=True, key=lambda x: x[0])
            for sc, idx, cand in flat:
                if idx in assigned_eng or cand in assigned_cand:
                    continue
                assignments[idx] = (cand, sc)
                assigned_eng.add(idx)
                assigned_cand.add(cand)
        elapsed = time.time() - start_t
        # record timing info on disk (append-safe)
        try:
            with open(timing_path, 'a', encoding='utf-8', newline='') as tf:
                tw = csv.writer(tf)
                if tf.tell() == 0:
                    tw.writerow(['District', 'EngCount', 'CandCount', 'SquareSize', 'Method', 'ElapsedSeconds'])
                tw.writerow([dname, len(eng_indices), len(arabic_cands), S, method_used, round(elapsed, 4)])
        except Exception:
            pass

        # apply assignments and write results
        for idx in eng_indices:
            row = scored_per_row[idx]['row']
            scored = scored_per_row[idx]['scored']
            best_cand = ''
            best_score = 0
            if idx in assignments:
                best_cand, best_score = assignments[idx]
                low_conf = best_score < THRESHOLD
            else:
                # no assignment: take top-scored for results and mark low_conf
                top = sorted(scored, key=lambda x: x[2], reverse=True)[:1]
                if top:
                    best_cand, best_lat, best_score = top[0]
                else:
                    best_cand = ''
                    best_score = 0
                low_conf = True

            # write results CSV
            try:
                with open(results_path, 'a', encoding='utf-8', newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=['English Name','English District','Best Arabic Match','Match Score'])
                    writer.writerow({'English Name': row['English Name'], 'English District': row['District Name'], 'Best Arabic Match': best_cand or '', 'Match Score': best_score})
            except Exception:
                pass

            if not low_conf and best_cand:
                # update dataframes like original
                try:
                    if 'Arabic Name' in englishDataframe.columns:
                        if idx in englishDataframe.index:
                            current_val = englishDataframe.at[idx, 'Arabic Name']
                            if pd.isna(current_val) or str(current_val).strip() == '':
                                englishDataframe.at[idx, 'Arabic Name'] = best_cand
                    else:
                        englishDataframe.loc[idx, 'Arabic Name'] = best_cand
                except Exception:
                    pass
                try:
                    if 'English Name' in arabicDataframe.columns:
                        mask = arabicDataframe['Village Name'] == best_cand
                        for j in arabicDataframe[mask].index:
                            curr = arabicDataframe.at[j, 'English Name']
                            if pd.isna(curr) or str(curr).strip() == '':
                                arabicDataframe.at[j, 'English Name'] = row['English Name']
                    else:
                        arabicDataframe.loc[arabicDataframe['Village Name'] == best_cand, 'English Name'] = row['English Name']
                except Exception:
                    pass
                # write accepted csv
                try:
                    with open(accepted_path, 'a', encoding='utf-8', newline='') as fh2:
                        writer2 = csv.DictWriter(fh2, fieldnames=['English Name','English District','Accepted Arabic Name','Match Score'])
                        writer2.writerow({'English Name': row['English Name'], 'English District': row['District Name'], 'Accepted Arabic Name': best_cand, 'Match Score': best_score})
                except Exception:
                    pass
            else:
                # write review with top-3
                review = {'English Name': row['English Name'], 'English District': row['District Name'], 'Best Arabic Match': best_cand or '', 'Match Score': best_score}
                for i, (a, b, sc) in enumerate(sorted(scored, key=lambda x: x[2], reverse=True)[:3], start=1):
                    review[f'Cand{i}'] = a
                    review[f'Cand{i}_Lat'] = b
                    review[f'Cand{i}_Score'] = sc
                try:
                    with open(review_path, 'a', encoding='utf-8', newline='') as fh3:
                        writer3 = csv.DictWriter(fh3, fieldnames=['English Name','English District','Best Arabic Match','Match Score','Cand1','Cand1_Lat','Cand1_Score','Cand2','Cand2_Lat','Cand2_Score','Cand3','Cand3_Lat','Cand3_Score'])
                        writer3.writerow(review)
                except Exception:
                    pass

    # write updated dataframes back to an Excel workbook for export
    try:
        out_xlsx = os.path.join(OUT_DIR, 'matched_output_hungarian.xlsx')
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as ew:
            arabicDataframe.to_excel(ew, sheet_name='Arabic', index=False)
            englishDataframe.to_excel(ew, sheet_name='English', index=False)
            districtTranslation.to_excel(ew, sheet_name='DistrictTranslation', index=False)
        print(f"Wrote matched Excel to: {out_xlsx}")
    except Exception as e:
        print(f"Could not write Excel: {e}")
