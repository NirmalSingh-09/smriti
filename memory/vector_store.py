import json
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parser.whatsapp_parser import parse_whatsapp_chat

def build_memory(chat_file, target_person, personality_file):
    print("\n🧠 Building Smriti memory bank...")

    with open(personality_file, 'r', encoding='utf-8') as f:
        personality = json.load(f)

    df = parse_whatsapp_chat(chat_file, target_person)
    messages = [m for m in df['message'].tolist() if len(m.split()) >= 3]

    print(f"📝 Storing {len(messages)} meaningful messages...")

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(messages)

    os.makedirs("memory", exist_ok=True)
    with open("memory/messages.pkl", "wb") as f:
        pickle.dump(messages, f)
    with open("memory/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("memory/matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    print(f"✅ Memory bank built — {len(messages)} messages stored!")
    return {"messages": messages, "vectorizer": vectorizer, "matrix": matrix}, personality


def search_memory(query, collection, n_results=3):
    messages = collection["messages"]
    vectorizer = collection["vectorizer"]
    matrix = collection["matrix"]

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[-n_results:][::-1]
    return [messages[i] for i in top_indices]