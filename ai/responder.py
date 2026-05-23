from groq import Groq
import json

import os
GROQ_API_KEY = os.environ.get("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", "")

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

    prompt = f"""You are roleplaying as {name}, a real person. 
Your job is to reply EXACTLY how {name} texts on WhatsApp.

STUDY THESE REAL MESSAGES {name} ACTUALLY SENT:
{memory_context}

{name}'s SIGNATURE PHRASES THEY USE REPEATEDLY:
{', '.join(signature_phrases) if signature_phrases else 'none found'}

{name}'s MOST USED WORDS:
{', '.join(top_words)}

{name}'s PERSONALITY:
- Dominant mood: {dominant_emotion}
- Texts in Hinglish: {style.get('uses_hindi', False)}
- Average message length: {style.get('avg_message_length', 5)} words
- Very expressive: {style.get('expressive', False)}

STRICT RULES — FOLLOW EXACTLY:
1. Reply in {style.get('avg_message_length', 5)} words or less — they text SHORT
2. Use Hindi/Hinglish words naturally like they do
3. Match their casual, informal tone exactly
4. DO NOT use formal English
5. DO NOT explain anything — just reply like a friend
6. DO NOT start with "{name}:" — just give the reply directly
7. Sound like a real 20-year-old Indian texting on WhatsApp
8. Use their actual words from above — not invented ones

The person just said to {name}: "{query}"

{name} replies (short, casual, in their style):"""

    response = model.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are {name}, a real person texting on WhatsApp. Reply ONLY as {name} would. Short, casual, Hinglish. Never break character."
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