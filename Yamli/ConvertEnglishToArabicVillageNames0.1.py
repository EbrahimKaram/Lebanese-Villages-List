#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import re ,sys, itertools
from VillageArray import *
reload(sys)
sys.setdefaultencoding('utf-8')

writeFile=open("Arabic_English_Pairs.txt","w")
##villages=["Aïn Abou Abdallah"]
i=0
for village in villages:
    villageModified=village.replace("-", "")
    listofWords=villageModified.split()
    arrayOfListOfArabicWords = []
    for phrase in listofWords:
        r = requests.get('https://api.yamli.com/transliterate.ashx?&tool=api&account_id=&prot=https%3A&hostname=www.yamli.com&path=%2Fapi%2Fsetup%2F&build=55155',
        data ={'word':phrase,'sxhr_id':str(4)})
        i=i+1
        response= r.text
        jsonString=response[62:-4]
        d=json.loads(jsonString)
        arabicSimilar=json.loads(d["data"])

        listOfArabicWords=re.split('(?:/[0-9][|])|(?:/[0-9])', arabicSimilar['r'])
        listOfArabicWords.pop()
        arrayOfListOfArabicWords.append(listOfArabicWords)
    ##print(arrayOfListOfArabicWords)
    listOfPhrases=list(itertools.product(*arrayOfListOfArabicWords))
    print(village)
    for phrase in listOfPhrases:
        ##print(" ".join(phrase))
        arabicPhrase=" ".join(phrase)
        if arabicPhrase in villagesArabic:
            print('\n')
            print ("Found Match")
            writeFile.write(arabicPhrase)
            writeFile.write(',')
            writeFile.write(village)
            writeFile.write('\n')
            print(arabicPhrase)
            print(village)
            break
writeFile.close()
