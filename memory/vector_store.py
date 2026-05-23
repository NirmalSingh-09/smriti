import chromadb
import json
import pandas as pd
from parser.whatsapp_parser import parse_whatsapp_chat

def build_memory(chat_file, target_person, personality_file):
    """
    Stores all of target person's messages in ChromaDB
    for semantic search later.
    """

    print("\n🧠 Building Smriti memory bank...")

    # Load personality profile
    with open(personality_file, 'r', encoding='utf-8') as f:
        personality = json.load(f)

    # Parse chat again
    df = parse_whatsapp_chat(chat_file, target_person)
    messages = df['message'].tolist()

    # Filter meaningful messages (more than 3 words)
    meaningful = [m for m in messages if len(m.split()) >= 3]
    print(f"📝 Storing {len(meaningful)} meaningful messages in memory...")

    # Setup ChromaDB (local, stores in smriti/memory/db folder)
    client = chromadb.PersistentClient(path="memory/db")

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("smriti_memory")
    except:
        pass

    collection = client.create_collection(
        name="smriti_memory",
        metadata={"hnsw:space": "cosine"}
    )

    # Store messages in batches
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


def search_memory(query, collection, n_results=5):
    """
    Finds most relevant messages from memory
    for a given query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results['documents'][0] if results['documents'] else []