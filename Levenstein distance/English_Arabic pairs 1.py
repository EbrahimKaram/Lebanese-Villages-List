from Levenstein_Distance import levenshtein
import pandas as pd
import requests
import codecs
import csv

arabicDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts.xlsx', sheet_name=0)
englishDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts.xlsx', sheet_name=1)

if __name__ == "__main__":
    # 1433 left
    # 1402
    englishNamesLeft = englishDataframe['English Name'][englishDataframe['Arabic Name'].isnull(
    )]

    # 905 length of names left
    arabicNamesLeft = arabicDataframe['Village Name'][arabicDataframe['English Name'].isnull(
    )]

    # Post requests to the following
    # https://glosbe.com/transliteration/api?from=Arabic&dest=Latin&text=%D8%B5%D9%88%D8%B1%D8%A7%D8%AA&format=json
    # for villageName in arabicNamesLeft:
    results = {}
 # TODO: fix english to araboc link up make sure names left are the same somewhat
 # TODO: json englihs names to multiple arabic if multiple pieces
    try:
        with codecs.open('conversions.csv', 'r', 'utf-8')as infile:
            reader = csv.reader(f)
            for english, arabic in reader:
                results[english]=arabic

    except :
        print("File doesn't exist")

    for englishName in englishNamesLeft:
        if not (englishName in results.keys()):
            payload = (('country_codes', 'lb'),
                    ('language', 'en'),
                    ('addressdetails', True), ('format', 'json'),
                    ('email', 'ebrahim_karam@hotmail.com'),
                    ('q', englishName),
                    ('country', 'Lebanon'),
                    ('limit', 1))
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search?", params=payload)

                if (r.text != '[]' and ('lb' in r.text)):
                    print(englishName)
                    arabicName=''
                    if('city' in r.text):
                        arabicName = r.json()[0]['address']['city']
                    elif ('village' in r.text):
                        arabicName = r.json()[0]['address']['village']
                    elif ('town' in r.text):
                        arabicName = r.json()[0]['address']['town']
                    elif ('county' in r.text):
                        arabicName = r.json()[0]['address']['county']
                    print(arabicName)
                    results[englishName]=arabicName

                    with codecs.open('conversions.csv', 'a+', 'utf-8')as outfile:
                        outfile.write("%s,%s\n" % (englishName, arabicName))

                    if arabicName in set(arabicNamesLeft):
                        englishNameRows = englishDataframe['English Name'] == englishName
                        englishDataframe.loc[englishNameRows,
                                            'Arabic Name'] = arabicName
                        print(englishDataframe[englishNameRows])
            except :
                print('Something went wrong')
