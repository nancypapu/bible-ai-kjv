from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import json, os, numpy as np, requests, threading
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
EMBED_FILE = "bible_embeddings.json"

SYSTEM_PROMPT = """
You are a Bible assistant.
Answer ONLY using the provided KJV verses.
Always cite Book Chapter:Verse.
"""

lock = threading.Lock()
embedding_status = {"ready": False, "progress": 0}

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_embeddings():
    global embedding_status

    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "r") as f:
            data = json.load(f)
        embedding_status["ready"] = True
        return

    embedding_status["ready"] = False
    embedding_status["progress"] = 0

    print("⬇️ Downloading KJV Bible...")
    resp = requests.get(KJV_URL)
    resp.raise_for_status()
    bible = json.loads(resp.content.decode("utf-8-sig"))

    verses = []
    for book in bible:
        for c_idx, chapter in enumerate(book["chapters"], start=1):
            for v_idx, text in enumerate(chapter, start=1):
                verses.append({
                    "book": book["name"],
                    "chapter": c_idx,
                    "verse": v_idx,
                    "text": text
                })

    total = len(verses)
    print(f"📖 Total verses: {total}")

    embeddings = []
    for i, v in enumerate(verses):
        emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=v["text"]
        )
        embeddings.append(emb.data[0].embedding)


        if i % 500 == 0:
            embedding_status["progress"] = int((i / total) * 100)
            print(f"Embedded {i}/{total}")

    with open(EMBED_FILE, "w") as f:
        json.dump({"verses": verses, "embeddings": embeddings}, f)

    embedding_status["ready"] = True
    embedding_status["progress"] = 100
    print("✅ Embeddings created successfully")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=build_embeddings, daemon=True).start()

class BibleQuery(BaseModel):
    question: str

@app.get("/status")
def status():
    return embedding_status

@app.post("/ask")
def ask_bible(q: BibleQuery):
    if not embedding_status["ready"]:
        return {
            "message": "Bible is still preparing. Please wait.",
            "progress": embedding_status["progress"]
        }

    with open(EMBED_FILE, "r") as f:
        data = json.load(f)

    verses = data["verses"]
    embeddings = np.array(data["embeddings"])

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
