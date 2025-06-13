import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load KB documents
with open("knowledge_base/projects.json") as f:
    projects = json.load(f)
with open("knowledge_base/bio.md") as f:
    bio = f.read()

# Prepare corpus
documents = [bio] + [p["description"] for p in projects]
embeddings = EMBED_MODEL.encode(documents, convert_to_tensor=False)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))


def retrieve_docs(query, k=3):
    query_vec = EMBED_MODEL.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k)
    return [documents[i] for i in I[0]]


def call_groq(prompt, model="llama3-8b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Persona.AI, a helpful assistant who speaks on behalf of Mahrad."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"].strip()


def answer_query(query):
    context = "\n".join(retrieve_docs(query))
    prompt = f"""
Answer the following question using only the context below.
Context:
{context}

Question: {query}
Answer:
"""
    return call_groq(prompt)


