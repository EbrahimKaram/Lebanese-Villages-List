from glob import glob
import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz import fuzz
import csv
import os


if __name__ == "__main__":

    OUT_DIR = os.path.dirname(__file__)

    names_of_excel = glob("English List*.xlsx")
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

    full_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                       .notnull()]
    full_info_english = englishDataframe[englishDataframe['Arabic Name']
                                         .isnull()]

    missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                            .isnull()]

    missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                          .isnull()]

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

        THRESHOLD = 60

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
            candidates = list(dict.fromkeys(possible_arabic_names))  # deduplicate preserve order
            scored = []
            for cand in candidates:
                try:
                    cand_lat = unidecode(str(cand)).strip()
                except Exception:
                    cand_lat = str(cand).strip()
                # score both original and stripped variants, keep the best
                s_orig = fuzz.token_set_ratio(english_name.lower(), cand_lat.lower())
                if english_name_stripped and english_name_stripped.lower() != english_name.lower():
                    s_stripped = fuzz.token_set_ratio(english_name_stripped.lower(), cand_lat.lower())
                    s = max(s_orig, s_stripped)
                else:
                    s = s_orig
                scored.append((cand, cand_lat, s))

            # pick best candidate
            scored.sort(key=lambda x: x[2], reverse=True)
            best_match, best_lat, score = scored[0]

            # consider low confidence if below threshold
            low_conf = score < THRESHOLD

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
        if (best_match is not None) and (best_match is not low_conf):
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
