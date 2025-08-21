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
#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":

    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory: {OUT_DIR}")
    # Set to True to see Yamli usage and other verbose debug prints
    VERBOSE = True
    names_of_excel = glob(os.path.join(OUT_DIR, "matched_output(178).xlsx"))
    print(names_of_excel)
    # if excel list is empty return
    if not names_of_excel:
        print("No Excel files found.")
        exit()

    arabicDataframe = pd.read_excel(names_of_excel[0], sheet_name=0)
    englishDataframe = pd.read_excel(names_of_excel[0], sheet_name=1)
    districtTranslation = pd.read_excel(names_of_excel[0], sheet_name=2)

    print("Arabic DataFrame:")
    print(arabicDataframe.head())
    print(arabicDataframe.columns)
    print("English DataFrame:")
    print(englishDataframe.head())
    print(englishDataframe.columns)
    print("District Translation DataFrame:")
    print(districtTranslation.head())
    print(districtTranslation.columns)

    # We will rely on unidecode transliteration for scoring. No custom mapping.
    mapping = {}

    # Yamli transliteration helper: returns list of Arabic candidate strings for an input latin word/phrase
    def get_yamli_candidates(text):
        # New Yamli API endpoint format — build params explicitly
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
            if VERBOSE:
                print(f"Yamli raw response length={len(raw)} for word='{text}'")
                # Yamli wraps the payload in a JS callback: Yamli.I.SXHRData.dataCallback({...});
                m = re.search(r"Yamli\.I\.SXHRData\.dataCallback\((\{.*\})\);", raw, re.S)
                if not m:
                    if VERBOSE:
                        print(f"Yamli: no JSON payload found for '{text}'")
                    # fallback: try to find first brace
                    idx = raw.find('{')
                    if idx == -1:
                        return []
                    payload = raw[idx:]
                else:
                    payload = m.group(1)

                # try to parse payload JSON robustly
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
                            if VERBOSE:
                                print(f"Yamli: unable to parse JSON payload for '{text}'")
                            # Fallback: try to extract Arabic script sequences directly from the raw text
                            arabic_seqs = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-]{2,}", raw)
                            items = []
                            for s in arabic_seqs:
                                s2 = s.strip()
                                if s2 and s2 not in items:
                                    items.append(s2)
                            if items:
                                if VERBOSE:
                                    print(f"Yamli: extracted {len(items)} Arabic sequences as fallback for '{text}'")
                                return items
                            return []

                # Some Yamli responses nest JSON inside 'data' as a string
                raw_data = j.get('data') if isinstance(j, dict) else None
            data = {}
            if isinstance(raw_data, str):
                try:
                    data = json.loads(raw_data)
                except Exception:
                    try:
                        data = ast.literal_eval(raw_data)
                    except Exception:
                        # fallback: try to extract an r= or 'r': value with regex
                        m = re.search(r"r\s*[:=]\s*['\"]([^'\"]+)['\"]", raw_data)
                        if m:
                            data = {'r': m.group(1)}
                        else:
                            data = {}
            elif isinstance(raw_data, dict):
                data = raw_data
            r = data.get('r', '')
            if not r:
                if VERBOSE:
                    print(f"Yamli: 'r' field empty for '{text}'")
                return []
            # split the r string into candidates (reuse existing regex)
            items = re.split(r'(?:/[0-9][|])|(?:/[0-9])', r)
            # drop empty and trailing elements
            items = [s for s in items if s]
            return items
        except Exception as e:
            if VERBOSE:
                print(f"Yamli call error for '{text}': {e}")
            return []

    # Helper to fetch a single token using a Session (used for parallel prefetch)
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
            # Yamli wraps the payload in a JS callback: Yamli.I.SXHRData.dataCallback({...});
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
                        # fallback to extracting Arabic sequences
                        arabic_seqs = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-]{2,}", payload)
                        items = []
                        for s in arabic_seqs:
                            s2 = s.strip()
                            if s2 and s2 not in items:
                                items.append(s2)
                        if items:
                            return items
                        return []

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
    full_info_english = englishDataframe[englishDataframe['Arabic Name']
                                         .isnull()]

    missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                            .isnull()]

    missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                          .isnull()]

    # prepare a set of known Arabic village names to validate matches
    arabic_village_set = set(missing_info_arabic['Village Name'].dropna().astype(str).tolist())
    ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    # track Arabic names that have already been paired to enforce one-to-one mapping
    used_arabic_set = set()

    # locality tokens to drop from Arabic phrases for matching (common noise)
    _LOCALITY_TOKENS = set(['وادي', 'واديه', 'واديّ', 'حي', 'حيه', 'حيّ', 'ال', 'منارة', 'منار', 'منطقه', 'منطقة'])

    def remove_locality_tokens(s: str) -> str:
        """Remove common locality tokens and leading 'ال' from an Arabic phrase and return a cleaned phrase."""
        if not s:
            return ''
        # split on whitespace and common dash characters
        parts = re.split(r'[\s\-\–\—]+', s)
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # remove leading Arabic article 'ال' for the token comparison
            p_noal = re.sub(r'^ال', '', p)
            # if the cleaned token is a locality token, skip it
            if p_noal in _LOCALITY_TOKENS or p in _LOCALITY_TOKENS:
                continue
            out.append(p_noal)
        return ' '.join(out)

    # --- Prepare Yamli token cache and parallel prefetch ---
    cache_path = os.path.join(OUT_DIR, 'yamli_token_cache.json')
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as cf:
                yamli_cache = json.load(cf)
        else:
            yamli_cache = {}
    except Exception:
        yamli_cache = {}

    # build unique token set from all English names to be processed
    # skip common article tokens (we don't want to call Yamli for 'el'/'al')
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

    if unique_tokens:
        if VERBOSE:
            print(f"Prefetching Yamli for {len(unique_tokens)} unique tokens...")
        # configure a session with retries
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 504))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        # parallel fetch
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
        # save cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as cf:
                json.dump(yamli_cache, cf, ensure_ascii=False, indent=1)
        except Exception:
            if VERBOSE:
                print(f"Could not write Yamli cache to {cache_path}")

    # iterate rows
    for index, row in missing_info_english.iterrows():

        # I want to get the list of possible Arabic names for this English name
        # I need to get the subset of the data frame that are in the same Mohfaza and district
        # districtTranslation gives me the translation from the arabic district name to the arabic district name
        # I need to get the arabic district name for that row and then find the possible arabic names from the missing arabic names

        arabic_district_name = districtTranslation.loc[
            districtTranslation['District'] == row['District Name'],
            'District Arabic'
        ].dropna().unique().tolist()

        # get a deduplicated list of Arabic district names matching the English district
        arabic_districts = districtTranslation.loc[
            districtTranslation['District'] == row['District Name'],
            'District Arabic'
        ].dropna().unique().tolist()

        if not arabic_districts:
            print(f"No Arabic district found for '{row['District Name']}' (English). Skipping.")
            continue

        # find possible Arabic village names where the District Name is in the matched arabic districts
        possible_arabic_names = missing_info_arabic[missing_info_arabic['District Name'].isin(arabic_districts)]['Village Name'].tolist()

        THRESHOLD = 80

        if not possible_arabic_names:
            # no candidates
            best_match = None
            score = 0
            low_conf = True
        else:
            # transliterate Arabics to Latin and compute fuzzy similarity with the English name
            english_name = str(row['English Name']).strip()
            # also create a variant with leading 'al'/'el' stripped (common Arabic article)
            english_name_stripped = re.sub(r'^(al|el)[\s\-_:]+', '', english_name, flags=re.I)
            # normalize hyphens/underscores to spaces for fuzzy matching (handles hyphenated names)
            english_norm = re.sub(r'[-_]', ' ', english_name).lower()
            english_stripped_norm = re.sub(r'[-_]', ' ', english_name_stripped).lower()
            # Deduplicated list of Arabic village names from the Arabic dataframe
            candidates = list(dict.fromkeys(possible_arabic_names))  # preserve order

            # Build Yamli per-token Arabic phrases (as before)
            yamli_phrases = []
            try:
                yamli_used = False
                tokens = re.sub(r'[-_]', ' ', english_name).split()
                if tokens:
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
                        per_token_lists.append(yc if yc else [t])
                    yamli_phrases = [' '.join(p) for p in itertools.product(*per_token_lists)]
                    if VERBOSE:
                        print(f"Yamli per-token candidates generated for '{english_name}': {len(yamli_phrases)}")
                if VERBOSE and not yamli_used:
                    print(f"Yamli returned no candidates for '{english_name}' (per-token only).")
            except Exception:
                if VERBOSE:
                    print(f"Yamli call failed for '{english_name}'")
                yamli_phrases = []

            # Now score each Arabic dataframe candidate by comparing it to Yamli Arabic suggestions
            scored = []
            if yamli_phrases:
                # prefer matching Arabic-script yamli suggestions
                for a_cand in candidates:
                    a_str = str(a_cand).strip()
                    a_stripped = re.sub(r'^\s*ال[\s\-_:]*', '', a_str)
                    best_score = 0
                    best_y = ''
                    for y in yamli_phrases:
                        if not ARABIC_RE.search(str(y)):
                            # skip non-Arabic yamli tokens
                            continue
                        y_str = str(y).strip()
                        y_stripped = re.sub(r'^\s*ال[\s\-_:]*', '', y_str)
                        try:
                            # compare original forms
                            s1 = fuzz.token_set_ratio(a_str, y_str)
                            # compare locality-stripped variations as well
                            a_noloc = remove_locality_tokens(a_str)
                            y_noloc = remove_locality_tokens(y_str)
                            s1_loc = fuzz.token_set_ratio(a_noloc, y_noloc) if a_noloc and y_noloc else 0
                            # compare stripped variations
                            s2 = fuzz.token_set_ratio(a_stripped, y_str) if a_stripped != a_str else 0
                            s3 = fuzz.token_set_ratio(a_str, y_stripped) if y_stripped != y_str else 0
                            s4 = fuzz.token_set_ratio(a_stripped, y_stripped) if (a_stripped != a_str and y_stripped != y_str) else 0
                            s = max(s1, s1_loc, s2, s3, s4)
                        except Exception:
                            s = 0
                        if s > best_score:
                            best_score = s
                            best_y = y_str
                    # if Yamli produced no Arabic suggestions that matched, we may fallback to transliteration vs English
                    if best_score == 0:
                        try:
                            cand_lat = unidecode(a_str)
                            # also try locality-stripped arabic candidate transliteration
                            cand_noloc = remove_locality_tokens(a_str)
                            cand_lat_noloc = unidecode(cand_noloc) if cand_noloc else ''
                            best_score = max(
                                fuzz.token_set_ratio(english_norm, cand_lat.lower()),
                                fuzz.token_set_ratio(english_norm, cand_lat_noloc.lower()) if cand_lat_noloc else 0
                            )
                        except Exception:
                            best_score = 0
                    scored.append((a_cand, best_y, best_score))
            else:
                # No Yamli suggestions available: fallback to previous unidecode transliteration scoring
                for cand in candidates:
                    try:
                        cand_str = str(cand).strip()
                        cand_stripped = re.sub(r'^\s*ال[\s\-_:]*', '', cand_str)
                        cand_lat = unidecode(cand_str)
                        cand_lat_stripped = unidecode(cand_stripped)
                    except Exception:
                        cand_lat = cand_str
                        cand_lat_stripped = cand_str
                    scores = []
                    s_orig = fuzz.token_set_ratio(english_norm, cand_lat.lower())
                    scores.append(s_orig)
                    if cand_lat_stripped != cand_lat:
                        s_arabic_stripped = fuzz.token_set_ratio(english_norm, cand_lat_stripped.lower())
                        scores.append(s_arabic_stripped)
                    # try locality-stripped candidate transliteration
                    cand_noloc = remove_locality_tokens(cand_str)
                    if cand_noloc and cand_noloc != cand_str:
                        try:
                            cand_lat_noloc = unidecode(cand_noloc)
                            scores.append(fuzz.token_set_ratio(english_norm, cand_lat_noloc.lower()))
                        except Exception:
                            pass
                    if english_name_stripped and english_name_stripped.lower() != english_name.lower():
                        s_english_stripped = fuzz.token_set_ratio(english_stripped_norm, cand_lat.lower())
                        scores.append(s_english_stripped)
                        if cand_lat_stripped != cand_lat:
                            s_both_stripped = fuzz.token_set_ratio(english_stripped_norm, cand_lat_stripped.lower())
                            scores.append(s_both_stripped)
                    s = max(scores) if scores else 0
                    scored.append((cand, cand_lat, s))

            # pick best candidate (only accept a candidate that exists in the Arabic dataframe)
            scored.sort(key=lambda x: x[2], reverse=True)
            best_match = None
            best_lat = None
            score = 0

            # Prefer highest-scoring candidate that is present in the Arabic dataframe
            chosen_in_arabic = None
            for cand, lat, sc in scored:
                # compare raw candidate string to entries in the arabic_village_set
                if str(cand) in arabic_village_set:
                    chosen_in_arabic = (cand, lat, sc)
                    break

            if chosen_in_arabic:
                # We found a candidate that exists in the Arabic dataframe.
                # Ensure it hasn't already been assigned to another English row.
                cand, lat, sc = chosen_in_arabic
                if cand in used_arabic_set:
                    # Try to find the next best Arabic-dataframe candidate that isn't used yet
                    next_choice = None
                    for cand2, lat2, sc2 in scored:
                        if str(cand2) in arabic_village_set and (str(cand2) not in used_arabic_set):
                            next_choice = (cand2, lat2, sc2)
                            break
                    if next_choice:
                        best_match, best_lat, score = next_choice
                        low_conf = score < THRESHOLD
                    else:
                        # No available Arabic candidate left — send to review
                        best_match = None
                        best_lat = None
                        score = 0
                        low_conf = True
                else:
                    best_match, best_lat, score = cand, lat, sc
                    low_conf = score < THRESHOLD
            else:
                # No candidate from the Arabic dataframe was found.
                # Do NOT accept Yamli-generated or arbitrary non-listed candidates.
                best_match = None
                best_lat = None
                score = 0
                low_conf = True

            # print only low-confidence results for manual inspection
            if low_conf:
                topk = scored[:5]
                print(f"Low-confidence matches for '{english_name}' (best_score={score}): ")
                for a, b, sc in topk:
                    print(f"  {a}  -> {b}  (score={sc})")

        # write result row to full results CSV for later review
        out_row = {
            'English Name': row['English Name'],
            'English District': row['District Name'],
            'Best Arabic Match': best_match if best_match is not None else '',
            'Match Score': score
        }
        with open(os.path.join(OUT_DIR, 'arabic_match_results.csv'), 'a', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['English Name', 'English District', 'Best Arabic Match', 'Match Score'])
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerow(out_row)

        # if confident, write to accepted CSV
        if (best_match is not None) and (not low_conf):
            # write accepted match back into the original dataframes
            try:
                # update the English dataframe row (index is from the original englishDataframe)
                # only set if the cell is empty/NaN to avoid overwriting existing values
                if 'Arabic Name' in englishDataframe.columns:
                    if index in englishDataframe.index:
                        current_val = englishDataframe.at[index, 'Arabic Name']
                        if pd.isna(current_val) or str(current_val).strip() == '':
                            englishDataframe.at[index, 'Arabic Name'] = best_match
                else:
                    # column missing: create it and set the value
                    englishDataframe.loc[index, 'Arabic Name'] = best_match
            except Exception:
                # if column/index issues occur, skip updating
                pass
            try:
                # update any matching rows in the Arabic dataframe to include the English name
                # only set where the 'English Name' cell is empty/NaN
                if 'English Name' in arabicDataframe.columns:
                    mask = arabicDataframe['Village Name'] == best_match
                    for j in arabicDataframe[mask].index:
                        curr = arabicDataframe.at[j, 'English Name']
                        if pd.isna(curr) or str(curr).strip() == '':
                            arabicDataframe.at[j, 'English Name'] = row['English Name']
                else:
                    # column missing: create it and set for matching rows
                    arabicDataframe.loc[arabicDataframe['Village Name'] == best_match, 'English Name'] = row['English Name']
                # reserve this Arabic name so it won't be paired again
                try:
                    used_arabic_set.add(best_match)
                except Exception:
                    pass
            except Exception:
                pass

            accepted = {
                'English Name': row['English Name'],
                'English District': row['District Name'],
                'Accepted Arabic Name': best_match,
                'Match Score': score
            }
            try:
                accepted_path = os.path.join(OUT_DIR, 'arabic_match_accepted.csv')
                with open(accepted_path, 'a', encoding='utf-8', newline='') as fh2:
                    writer2 = csv.DictWriter(fh2, fieldnames=['English Name', 'English District', 'Accepted Arabic Name', 'Match Score'])
                    if fh2.tell() == 0:
                        writer2.writeheader()
                    writer2.writerow(accepted)
            except PermissionError:
                # fallback to repo root if write in OUT_DIR is denied
                fallback_path = os.path.abspath(os.path.join(OUT_DIR, '..', 'arabic_match_accepted.csv'))
                print(f"Warning: cannot write to {accepted_path}; falling back to {fallback_path}")
                with open(fallback_path, 'a', encoding='utf-8', newline='') as fh2:
                    writer2 = csv.DictWriter(fh2, fieldnames=['English Name', 'English District', 'Accepted Arabic Name', 'Match Score'])
                    if fh2.tell() == 0:
                        writer2.writeheader()
                    writer2.writerow(accepted)

        # write review CSV for low-confidence rows (top-3 candidates)
        if low_conf:
            review = {
                'English Name': row['English Name'],
                'English District': row['District Name'],
                'Best Arabic Match': best_match if best_match is not None else '',
                'Match Score': score
            }
            # include top-3 if available
            if 'scored' in locals():
                for i, (a, b, sc) in enumerate(scored[:3], start=1):
                    review[f'Cand{i}'] = a
                    review[f'Cand{i}_Lat'] = b
                    review[f'Cand{i}_Score'] = sc
            else:
                # placeholders if no candidates
                for i in range(1, 4):
                    review[f'Cand{i}'] = ''
                    review[f'Cand{i}_Lat'] = ''
                    review[f'Cand{i}_Score'] = ''

            review_fields = ['English Name', 'English District', 'Best Arabic Match', 'Match Score',
                             'Cand1', 'Cand1_Lat', 'Cand1_Score',
                             'Cand2', 'Cand2_Lat', 'Cand2_Score',
                             'Cand3', 'Cand3_Lat', 'Cand3_Score']
            with open(os.path.join(OUT_DIR, 'arabic_match_review.csv'), 'a', encoding='utf-8', newline='') as fh3:
                writer3 = csv.DictWriter(fh3, fieldnames=review_fields)
                if fh3.tell() == 0:
                    writer3.writeheader()
                writer3.writerow(review)

    # after processing all rows, write updated dataframes back to an Excel workbook for export
    try:
        out_xlsx = os.path.join(OUT_DIR, 'matched_output.xlsx')
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as ew:
            arabicDataframe.to_excel(ew, sheet_name='Arabic', index=False)
            englishDataframe.to_excel(ew, sheet_name='English', index=False)
            districtTranslation.to_excel(ew, sheet_name='DistrictTranslation', index=False)

        print(f"Wrote matched Excel to: {out_xlsx}")
    except PermissionError:
        fallback_xlsx = os.path.abspath(os.path.join(OUT_DIR, '..', 'matched_output.xlsx'))
        with pd.ExcelWriter(fallback_xlsx, engine='openpyxl') as ew:
            arabicDataframe.to_excel(ew, sheet_name='Arabic', index=False)
            englishDataframe.to_excel(ew, sheet_name='English', index=False)
            districtTranslation.to_excel(ew, sheet_name='DistrictTranslation', index=False)
        print(f"Warning: could not write to OUT_DIR; wrote matched Excel to fallback: {fallback_xlsx}")
