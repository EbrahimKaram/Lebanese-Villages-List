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



# Future Attempts
Do a join for all the villages that are divided into multiple ones.
## Mechanical Turk
Improve the map but also do a blank submission for those that lack names.
You would get an email with the potential name.
This would be a way to crowd source the names.
## Arabic to IPA
If we compare the IPA of each village name, we might be able to get somewhere.
So arabic to IPA
latin to IPA
then compare the IPA's of both
