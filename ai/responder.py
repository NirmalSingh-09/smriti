from groq import Groq
import json

GROQ_API_KEY = "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"

def setup_gemini():
    client = Groq(api_key=GROQ_API_KEY)
    return client


def generate_response(query, relevant_memories, personality, model, history=None):

    top_words = list(personality.get('top_words', {}).keys())[:15]
    catchphrases = personality.get('catchphrases', [])[:8]
    style = personality.get('style', {})
    dominant_emotion = personality.get('dominant_emotion', 'humor')
    signature_phrases = personality.get('signature_phrases', [])[:5]
    name = personality.get('name', 'this person')

    memory_context = "\n".join([f"- {m}" for m in relevant_memories])

    history_context = ""
    if history:
        history_context = "\nRECENT CONVERSATION:\n" + "\n".join(history[-6:]) + "\n"

    prompt = f"""You are roleplaying as {name}, a real Indian person texting on WhatsApp.

THEIR PERSONALITY:
- Dominant mood: {dominant_emotion}
- Favourite words they use: {', '.join(top_words[:10])}
- Their catchphrases: {', '.join(catchphrases[:5])}
- Texts in Hinglish: {style.get('uses_hindi', True)}
- Average message length: {style.get('avg_message_length', 5)} words

SOME ACTUAL MESSAGES THEY SENT:
{memory_context}
{history_context}
STRICT RULES:
1. You are a real Indian person — reply like one
2. Reply in maximum {style.get('avg_message_length', 5)} words
3. Use Hinglish naturally — mix Hindi and English like Indians do on WhatsApp
4. If someone says "kya kar raha hai" reply like "kuch nahi yaar tu bta" or "bas timepass"
5. If someone says "khelega" reply like "haa chal" or "nahi yaar mood nahi"
6. NEVER say you don't understand — always reply naturally
7. NEVER use formal English
8. Sound like a 20 year old Indian on WhatsApp
9. Use their actual catchphrases and words from above
10. DO NOT repeat what they said back to them
11. Keep it real, casual, short

Person says: "{query}"

{name} replies instantly like on WhatsApp:"""

    response = model.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are {name}, a real Indian person texting on WhatsApp. You understand Hindi, English and Hinglish perfectly. Reply ONLY as {name} — short, casual, real. Never break character. Never say you don't understand Hindi."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=80,
        temperature=0.9
    )
    return response.choices[0].message.content.strip()