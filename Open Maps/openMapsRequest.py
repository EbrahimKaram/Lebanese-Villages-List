#  https://nominatim.openstreetmap.org/search?<params>
# https://nominatim.org/release-docs/develop/api/Search/
import requests

village_name = 'الكنيسة'
payload = (('country_codes', 'lb'),
           ('language', 'en'),
           ('addressdetails', True), ('format', 'json'),
           ('email','ebrahim_karam@hotmail.com'),
           ('city',village_name),
           ('country','Lebanon'),
           ('limit', 1))
r = requests.get("https://nominatim.openstreetmap.org/search?", params=payload)

if (r.text != '[]'):
    if('city' in r.text):
        print(r.json()[0]['address']['city'])
    elif ('village' in r.text):
        print(r.json()[0]['address']['village'])
    elif ('town' in r.text):
        print(r.json()[0]['address']['town'])
    print(r.text)

