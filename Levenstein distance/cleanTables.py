import pandas as pd

arabicDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts.xlsx',
    sheet_name=0)
englishDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts.xlsx',
    sheet_name=2)
full_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                   .notnull()]
full_info_english = englishDataframe[englishDataframe['Arabic Name']
                                     .isnull()]

missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                        .isnull()]

for index, row in full_info_arabic.iterrows():
    if row['English Name'] in set(missing_info_english['English Name']):
        englishDataframe
        englishNameRows = (
            englishDataframe['English Name'] == row['English Name'])
        englishDataframe.loc[englishNameRows,
                             'Arabic Name'] = row['Village Name']
        # print("found one")

full_info_english = englishDataframe[englishDataframe['Arabic Name']
                                     .notnull()]

missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                        .isnull()]

missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                      .isnull()]

for index, row in full_info_english.iterrows():
    if row['Arabic Name'] in set(missing_info_arabic['Village Name']):

        arabicNameRows = (
            arabicDataframe['Village Name'] == row['Arabic Name'])
        arabicDataframe.loc[arabicNameRows,
                            'English Name'] = row['English Name']
        print("found Arabic filler")

englishDataframe.to_csv('English Info.csv')
arabicDataframe.to_csv('Arabic Info.csv')