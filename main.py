from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import os
import numpy as np
import requests
import threading
import time

# ---------------- CONFIG ----------------

KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
EMBED_FILE = "bible_embeddings.json"

SYSTEM_PROMPT = """
You are a Bible assistant.
Answer ONLY using the provided KJV Bible verses.
Always include Book Chapter:Verse.
If the answer is not found in scripture, say so.
"""

# ---------------- FASTAPI APP ----------------

app = FastAPI()

embedding_status = {
    "ready": False,
    "progress": 0
}

# ---------------- OPENAI CLIENT (RETRY SAFE) ----------------

def get_openai_client(wait=True):
    while True:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)

        if not wait:
            raise RuntimeError("OPENAI_API_KEY is not set")

        print("⏳ Waiting for OPENAI_API_KEY to be available...")
        time.sleep(2)

# ---------------- UTILS ----------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- EMBEDDING BUILDER ----------------

def build_embeddings():
    global embedding_status

    # If embeddings already exist, skip
    if os.path.exists(EMBED_FILE):
        embedding_status["ready"] = True
        embedding_status["progress"] = 100
        print("✅ Embeddings already exist")
        return

    print("⬇️ Downloading KJV Bible...")
    resp = requests.get(KJV_URL)
    resp.raise_for_status()

    # Handle UTF-8 BOM safely
    bible = json.loads(resp.content.decode("utf-8-sig"))

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

    total = len(verses)
    print(f"📖 Total verses: {total}")

    client = get_openai_client()
    embeddings = []

    print("🧠 Creating embeddings (one-time process)...")

    for i, verse in enumerate(verses):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=verse["text"]
        )
        embeddings.append(emb.data[0].embedding)

        if i % 500 == 0:
            embedding_status["progress"] = int((i / total) * 100)
            print(f"Embedded {i}/{total}")

    with open(EMBED_FILE, "w") as f:
        json.dump({
            "verses": verses,
            "embeddings": embeddings
        }, f)

    embedding_status["ready"] = True
    embedding_status["progress"] = 100
    print("✅ Embeddings created successfully")

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    threading.Thread(target=build_embeddings, daemon=True).start()

# ---------------- API MODELS ----------------

class BibleQuery(BaseModel):
    question: str

# ---------------- API ENDPOINTS ----------------

@app.get("/status")
def status():
    return embedding_status

@app.post("/ask")
def ask_bible(query: BibleQuery):
    if not embedding_status["ready"]:
        return {
            "message": "Bible is still preparing. Please wait.",
            "progress": embedding_status["progress"]
        }

    with open(EMBED_FILE, "r") as f:
        data = json.load(f)

    verses = data["verses"]
    embeddings = np.array(data["embeddings"])

    client = get_openai_client()

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query.question
    ).data[0].embedding

    scores = [cosine_similarity(q_emb, e) for e in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    context = "\n".join(
        f'{verses[i]["book"]} {verses[i]["chapter"]}:{verses[i]["verse"]} - {verses[i]["text"]}'
        for i in top_indices
    )

    answer = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
        ]
    )

    return {
        "answer": answer.choices[0].message.content,
        "verses": [verses[i] for i in top_indices]
    }
