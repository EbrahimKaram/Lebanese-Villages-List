import requests
import json
import re ,sys
from VillageArray import *
reload(sys)
sys.setdefaultencoding('utf-8')

writeFile=open("Arabic_English_Pairs.txt","w")

i=1
for village in villages:
    villageSend=village.replace("-", "")
    r = requests.get('https://api.yamli.com/transliterate.ashx?&tool=api&account_id=&prot=https%3A&hostname=www.yamli.com&path=%2Fapi%2Fsetup%2F&build=5515&sxhr_id=15',
    data ={'word':villageSend,'sxhr_id':str(15+i)})
    i=i+1
    response= r.text
    jsonString=response[62:-4]
    d=json.loads(jsonString)
    arabicSimilar=json.loads(d["data"])

    ##listOfArabicWords=arabicSimilar['r'][:-2].split("/2|")
    listOfArabicWords=re.split('(?:/[0-9][|])|(?:/[0-9])', arabicSimilar['r'])
    print(village)
    for word in listOfArabicWords:
        if word in villagesArabic:
            print ("Found Match")
            writeFile.write(word)
            writeFile.write(',')
            writeFile.write(village)
            writeFile.write('\n')
            print(word)
            print(village)
            break
writeFile.close()
