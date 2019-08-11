import pandas as pd
import requests

arabicDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts (Use this) .xlsx', sheet_name=0)
englishDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts (Use this) .xlsx', sheet_name=1)

missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                        .isnull()]

missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                      .isnull()]

missing_info_english.loc[:, 'openMapsData'] = ""

for index, village in missing_info_english.iterrows():
    print(index)
    village_name = village['English Name']
    payload = (('country_codes', 'lb'),
               ('language', 'en'),
               ('addressdetails', True), ('format', 'json'),
               ('email', 'ebrahim_karam@hotmail.com'),
               ('city', village_name),
               ('country', 'Lebanon'),
               ('limit', 1))
    r = requests.get(
        "https://nominatim.openstreetmap.org/search?", params=payload)
    if (r.text != '[]' and ('lb' in r.text)):
        missing_info_english['openMapsData'][index] = r.text
        
        print(village_name)
        print(r.json()[0]['address'])

missing_info_english.to_excel('OpenMapsDataEnglish.xlsx', sheet_name='English Villages')
