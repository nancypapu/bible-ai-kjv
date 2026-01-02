from fastapi import FastAPI
from pydantic import BaseModel
import openai, json, os, numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

with open("bible_embeddings.json", "r") as f:
    data = json.load(f)

verses = data["verses"]
embeddings = np.array(data["embeddings"])

class BibleQuery(BaseModel):
    question: str

SYSTEM_PROMPT = """
You are a Bible assistant.
Answer ONLY using the provided KJV verses.
Always cite Book Chapter:Verse.
"""

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

