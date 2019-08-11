from VillageArray import *
import sys, os
reload(sys)
sys.setdefaultencoding('utf-8')

for village in EnglishVillagesToRemove:
    if village in villages:
        ##print village
        villages.remove(village)
writeFile=open("EnglishVillages.txt",'w')
writeFile.write("villages = [")
for village in villages:
    writeFile.write("\""+village.strip()+"\",")
writeFile.seek(-1, os.SEEK_END)
writeFile.truncate()
writeFile.write("]")
writeFile.close
