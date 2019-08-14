import pandas as pd

arabicMunicipalities = pd.read_excel(
    'Liban Data\List of Municipalities.xlsx', sheet_name=0)
englishMunicipalities = pd.read_excel(
    'Liban Data\List of Municipalities.xlsx', sheet_name=1)

englishVillages = pd.read_excel(
    'English List of districts. Arabic names of districts 2.0.xlsx', sheet_name=1)

arabicVillages = pd.read_excel(
    'English List of districts. Arabic names of districts 2.0.xlsx', sheet_name=0)

missing_info_english = englishVillages[englishVillages['Arabic Name']
                                       .isnull()]

missing_info_arabic = arabicVillages[arabicVillages['English Name']
                                     .isnull()]

count = 0
for englishVillage in missing_info_english['English Name']:
    for index, englishMunicipality in englishMunicipalities.iterrows():
        if str(englishVillage) == str(englishMunicipality['Municipality']):
            arabicMunicipality = arabicMunicipalities.loc[index, [
                'البلدية']]['البلدية']
            if arabicMunicipality in set(missing_info_arabic['Village Name']):
                if arabicMunicipality != 'nan':
                    count += 1
                    print(englishVillage)
                    print(englishMunicipality['Municipality'])
                    print(arabicMunicipality)
                    # TODO start modifiying the bigger Data Frame
                    arabicNameRows = (
                        arabicVillages['Village Name'] == arabicMunicipality)
                    arabicVillages.loc[arabicNameRows,
                                    'English Name'] = englishVillage

                    englishNameRows = (
                        englishVillages['English Name'] == englishVillage)
                    englishVillages.loc[englishNameRows, 'Arabic Name'] = arabicMunicipality


with pd.ExcelWriter('hello.xlsx') as writer:
    englishVillages.to_excel(writer, sheet_name='English')
    arabicVillages.to_excel(writer, sheet_name='Arabic')
