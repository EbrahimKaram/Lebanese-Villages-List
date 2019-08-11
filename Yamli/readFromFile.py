import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')

FileToRead=open("List Of Arabic Districts.txt","r")
lines=FileToRead.readlines()
FileToRead.close()
##writeFile=open("ArrayOfArabic villages.txt",'w')
writeFile=open("Array of Arabic Disctrics.txt",'w')

writeFile.write("EnglishVillagesToRemove=[")
## writeFile.write("villagesArabic=[")
for line in lines:
    if (line!='\n'):
        writeFile.write("\""+line.strip()+"\",")

writeFile.seek(-1, os.SEEK_END)
writeFile.truncate()
writeFile.write("]")
writeFile.close
