import json
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("bible_kjv.json", "r", encoding="utf-8") as f:
    bible = json.load(f)

verses = []
for book in bible:
    book_name = book["name"]
    for c_idx, chapter in enumerate(book["chapters"], start=1):
        for v_idx, text in enumerate(chapter, start=1):
            verses.append({
                "book": book_name,
                "chapter": c_idx,
                "verse": v_idx,
                "text": text
            })

print(f"Total verses: {len(verses)}")

embeddings = []
for v in verses:
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=v["text"]
    )
    embeddings.append(emb.data[0].embedding)

with open("bible_embeddings.json", "w") as f:
    json.dump({
        "verses": verses,
        "embeddings": embeddings
    }, f)

print("✅ Embeddings saved")

