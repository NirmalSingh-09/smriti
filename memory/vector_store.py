import chromadb
import json
import pandas as pd
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from parser.whatsapp_parser import parse_whatsapp_chat

embedding_fn = ONNXMiniLM_L6_V2()

def build_memory(chat_file, target_person, personality_file):
    print("\n🧠 Building Smriti memory bank...")

    with open(personality_file, 'r', encoding='utf-8') as f:
        personality = json.load(f)

    df = parse_whatsapp_chat(chat_file, target_person)
    messages = df['message'].tolist()

    meaningful = [m for m in messages if len(m.split()) >= 3]
    print(f"📝 Storing {len(meaningful)} meaningful messages in memory...")

    client = chromadb.PersistentClient(path="memory/db")

    try:
        client.delete_collection("smriti_memory")
    except:
        pass

    collection = client.create_collection(
        name="smriti_memory",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn
    )

    batch_size = 100
    for i in range(0, len(meaningful), batch_size):
        batch = meaningful[i:i+batch_size]
        collection.add(
            documents=batch,
            ids=[f"msg_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": target_person} for _ in batch]
        )

    print(f"✅ Memory bank built — {len(meaningful)} messages stored!")
    print(f"📁 Saved to memory/db/")

    return collection, personality


def search_memory(query, collection, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []