# Introduction
This is repo is an attempt to map the Arabic village name to their Latin counterpart. These villages are villages in Lebanon.

The Arabic village lit was retrieved from an Official Lebanese government website. This is our master list.

The Latin names were extracted from JSON shape file located on the following site
https://gadm.org/download_country_v3.html

# Motives
This is to improve my D3 district map for Lebanon
https://ebrahimkaram.github.io/Lebanon-Districts-D3-Map/

# Attempts
I have been trying to map the Latin names with their Arabic counterparts for the sake of organizing Data. These attempts have used the following methods.
* Using the Yamli api on the Latin name and matching it with the following Arabic name
* Doing a Latin translation for the Arabic names and doing a Levenstein distance on every word.
* Using the Open Maps api (https://nominatim.openstreetmap.org/search?)
* Using the data from LibanData and doing matches based on that


# Future Attempts (If I'm still motivated)
Do a join for all the villages that are divided into multiple ones.
## Mechanical Turk
Make a WebApp that would allow people to add the arabic village name from the Lebanon D3 District map itself. 

Possible UI/Implemetation)
They click on the blank village name. They get diverted to maybe a Google Submit form with prefilled info. They would then select from a drown downlist of possible arabic names of what they think it is. 
I guess that might be easy (ish)

## Arabic to IPA
If we compare the IPA of each village name, we might be able to get somewhere.
So arabic to IPA
latin to IPA
then compare the IPA's of both. 

Might be a bit convoluted. 

## Other ideas
Probably perform a transliteration
Look into the following
https://www.geonames.org/export/ws-overview.html

# How to Contribute

If you are eager to help. Just download the repo and make a merge request. 
Follow the steps below when editing:
1. OpenEnglish List of districts. Arabic names of districts 2.0.xlsx
2. create a column with your name. 
3. Put a one on all the villages that you edited in the column with your name. 

Then Simply create a merge request



