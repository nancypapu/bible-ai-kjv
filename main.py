from fastapi import FastAPI
from pydantic import BaseModel
import openai, json, os, numpy as np, requests, time

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

EMBED_FILE = "bible_embeddings.json"
BIBLE_FILE = "bible_kjv.json"

KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

SYSTEM_PROMPT = """
You are a Bible assistant.
Answer ONLY using the provided KJV verses.
Always cite Book Chapter:Verse.
If not found, say so.
"""

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def setup_data_if_needed():
    if os.path.exists(EMBED_FILE):
        print("✅ Embeddings already exist")
        return

    print("⬇️ Downloading KJV Bible...")
    resp = requests.get(KJV_URL)
    resp.raise_for_status()
    bible = resp.json()

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

    print(f"📖 Total verses: {len(verses)}")
    print("🧠 Creating embeddings (one-time, may take a few minutes)...")

    embeddings = []
    for i, v in enumerate(verses):
        emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=v["text"]
        )
        embeddings.append(emb.data[0].embedding)

        if i % 1000 == 0:
            print(f"Embedded {i}/{len(verses)} verses")

    with open(EMBED_FILE, "w") as f:
        json.dump({"verses": verses, "embeddings": embeddings}, f)

    print("✅ Embeddings created successfully")

# 🔥 Run setup ONCE at startup
setup_data_if_needed()

with open(EMBED_FILE, "r") as f:
    data = json.load(f)

verses = data["verses"]
embeddings = np.array(data["embeddings"])

class BibleQuery(BaseModel):
    question: str

@app.post("/ask")
def ask_bible(q: BibleQuery):
    q_emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=q.question
    ).data[0].embedding

    scores = [cosine(q_emb, e) for e in embeddings]
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    context = "\n".join(
        f'{verses[i]["book"]} {verses[i]["chapter"]}:{verses[i]["verse"]} - {verses[i]["text"]}'
        for i in top
    )

    ans = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.question}"}
        ]
    )

    return {
        "answer": ans.choices[0].message.content,
        "verses": [verses[i] for i in top]
    }
