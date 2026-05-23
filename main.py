from parser.whatsapp_parser import parse_whatsapp_chat
from personality.extractor import extract_personality, print_personality_report
from memory.vector_store import build_memory, search_memory
from ai.responder import setup_gemini, generate_response
import json
import os

# ── CONFIG ─────────────────────────────────────────────
CHAT_FILE = "data/chat.txt"
TARGET_PERSON = "Kinder Joy"
PERSONALITY_FILE = "data/personality_profile.json"

if __name__ == "__main__":
    print("\n🌸 SMRITI — Waking up memories...\n")

    # Load personality
    with open(PERSONALITY_FILE, 'r', encoding='utf-8') as f:
        personality = json.load(f)

    # Build or load memory
    if not os.path.exists("memory/db"):
        collection, personality = build_memory(
            CHAT_FILE, TARGET_PERSON, PERSONALITY_FILE
        )
    else:
        import chromadb
        client = chromadb.PersistentClient(path="memory/db")
        collection = client.get_collection("smriti_memory")
        print("✅ Memory loaded from database!")

    # Setup AI
    model = setup_gemini()
    print("✅ AI connected!")

    print(f"\n💬 You can now talk to {TARGET_PERSON}")
    print("   Type 'quit' to exit\n")
    print("="*50)

    # Conversation history
    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print(f"\n🌸 Goodbye. {TARGET_PERSON}'s memories are safe.\n")
            break

        if not user_input:
            continue

        # Add to history
        conversation_history.append(f"You: {user_input}")

        # Search relevant memories
        memories = search_memory(user_input, collection, n_results=3)

        # Generate response with history
        response = generate_response(
            user_input,
            memories,
            personality,
            model,
            conversation_history[-6:]  # last 3 exchanges
        )

        # Add response to history
        conversation_history.append(f"{TARGET_PERSON}: {response}")

        print(f"\n{TARGET_PERSON}: {response}")