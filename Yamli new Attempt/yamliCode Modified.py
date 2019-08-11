#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import re
import sys
import itertools
import pandas as pd


arabicDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts (Use this) .xlsx', sheet_name=0)
arabicDataframe['Village Name'] = arabicDataframe['Village Name'].str.strip()
englishDataframe = pd.read_excel(
    'English List of districts. Arabic names of districts (Use this) .xlsx', sheet_name=1)

missing_info_english = englishDataframe[englishDataframe['Arabic Name']
                                        .isnull()]

missing_info_arabic = arabicDataframe[arabicDataframe['English Name']
                                      .isnull()]


for village in missing_info_english['English Name']:
    villageModified = village.replace("-", " ")
    villageModified = villageModified.lower()
    villageModified = villageModified.strip()
    if 'el' in villageModified:
        villageList = list(villageModified)
        index = villageModified.index('el')+2
        if (index<len(villageList)):
            if (villageList[index]==" "):
                villageList[index] = ""
        villageModified = "".join(villageList)
    if 'al' in villageModified:
        villageList = list(villageModified)
        index = villageModified.index('al')+2
        if (index < len(villageList)):
            if (villageList[villageModified.index('al')+2] == " "):
                villageList[villageModified.index('al')+2] = ""
        villageModified = "".join(villageList)

    listofWords = villageModified.split()
    if(len(listofWords)<2):
        listofWords.append('el'+villageModified)
        listofWords.append(villageModified.replace("el", ""))
        listofWords.append(villageModified.replace("al", ""))
    arrayOfListOfArabicWords = []
    for phrase in listofWords:
        phrase=phrase.strip()
        r = requests.get('https://api.yamli.com/transliterate.ashx?&tool=api&account_id=&prot=https%3A&hostname=www.yamli.com&path=%2Fapi%2Fsetup%2F&build=55155',
                         data={'word': phrase, 'sxhr_id': str(4)})
        response = r.text
        jsonString = response[62:-4]
        d = json.loads(jsonString)
        arabicSimilar = json.loads(d["data"])

        listOfArabicWords = re.split(
            '(?:/[0-9][|])|(?:/[0-9])', arabicSimilar['r']) +[""]
        listOfArabicWords.pop()
        arrayOfListOfArabicWords.append(listOfArabicWords)
    # print(arrayOfListOfArabicWords)
    listOfPhrases = list(itertools.product(*arrayOfListOfArabicWords))
    print(village)
    for phrase in listOfPhrases:
        ##print(" ".join(phrase))
        arabicPhrase = " ".join(phrase)
        if arabicPhrase in set(missing_info_arabic['Village Name']):
            print('\n')
            print("Found Match")
            arabicNameRows = (arabicDataframe['Village Name'] == arabicPhrase)
            arabicDataframe.loc[arabicNameRows,
                                'English Name'] = village

            englishNameRows = (englishDataframe['English Name'] == village)
            englishDataframe.loc[englishNameRows, 'Arabic Name'] = arabicPhrase
            print(arabicPhrase)
            print(village)
            break

