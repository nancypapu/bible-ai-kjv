import requests
import json

URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

print("Downloading KJV Bible...")
resp = requests.get(URL)
resp.raise_for_status()

with open("bible_kjv.json", "w", encoding="utf-8") as f:
    json.dump(resp.json(), f, ensure_ascii=False)

print("✅ KJV Bible downloaded")

