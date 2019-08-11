from Levenstein_Distance import levenshtein
import pandas as pd
import requests
import codecs
import csv

arabicDataframe=pd.read_excel('English List of districts. Arabic names of districts.xlsx',sheet_name=0)
englishDataframe=pd.read_excel('English List of districts. Arabic names of districts.xlsx',sheet_name=1)

if __name__ == "__main__":
    ##1433 left
    englishNamesLeft=englishDataframe['English Name'][englishDataframe['Arabic Name'].isnull()]
    
    ## 905 length of names left
    arabicNamesLeft=arabicDataframe['Village Name'][arabicDataframe['English Name'].isnull()]

    ## Post requests to the following
    # https://glosbe.com/transliteration/api?from=Arabic&dest=Latin&text=%D8%B5%D9%88%D8%B1%D8%A7%D8%AA&format=json
    # for villageName in arabicNamesLeft:
    
    arabic_latin_pairs={}
    with codecs.open('latinPairs.csv', 'r','utf-8')as infile:
        reader = csv.reader(infile)
        arabic_latin_pairs = {rows[0]:rows[1] for rows in reader}

    for village_name in arabicNamesLeft:
    # village_name='التبانة'
        payload = (('from', 'Arabic'), ('dest', 'Latin'),('format','json'),('text',village_name))
        if village_name not in arabic_latin_pairs.keys():
            r = requests.get("https://glosbe.com/transliteration/api", params=payload)
            latin_name=r.json()['text']
            print(village_name)
            print(latin_name)
            arabic_latin_pairs[village_name]=latin_name

    with codecs.open('latinPairs.csv', 'w','utf-8') as f:
        for key in arabic_latin_pairs.keys():
            f.write("%s,%s\n"%(key,arabic_latin_pairs[key]))
## There are only 893 unique items. so 13 are duplicates

# TODO: go over list of latin names and find its distance with the every english name. Then the min should be the english word for it.
for key in arabic_latin_pairs.keys():
    villages=[]
    distances=[]
    for englishVillage in englishNamesLeft:
        distance=levenshtein(arabic_latin_pairs[key],englishVillage)
        villages.append(englishVillage)
        distances.append(distance)
    
    min_village=villages[distances.index(min(distances))]

    if (len(min_village)<len(key)+3) and ((len(min_village)>len(key)-3)):
        print(min(distances))   
        print(min_village)
        print(key)
    
    
        