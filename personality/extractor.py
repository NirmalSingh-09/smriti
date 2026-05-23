import pandas as pd
from collections import Counter
import re

def extract_personality(df: pd.DataFrame, target_name: str) -> dict:
    """
    Extracts personality profile from parsed messages.
    Returns a rich personality dictionary.
    """

    messages = df['message'].tolist()
    full_text = ' '.join(messages)
    words = full_text.lower().split()

    personality = {}

    # ── 1. VOCABULARY FINGERPRINT ──────────────────────────────
    # Most used unique words (excluding common stopwords)
    stopwords = set(['the','a','an','is','it','in','on','at','to',
                     'and','or','but','i','you','we','he','she','they',
                     'was','are','be','been','have','has','had','do',
                     'did','will','would','could','should','of','for',
                     'this','that','with','my','your','me','him','her',
                     'so','just','not','no','yes','ok','okay','hi','hello'])

    meaningful_words = [w for w in words if w not in stopwords
                        and len(w) > 2 and w.isalpha()]
    vocab_counter = Counter(meaningful_words)
    personality['top_words'] = vocab_counter.most_common(30)

    # ── 2. CATCHPHRASES (2-3 word combos used repeatedly) ──────
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    bigram_counter = Counter(bigrams)

    # Only phrases used 3+ times are real catchphrases
    personality['catchphrases'] = [
        phrase for phrase, count in bigram_counter.most_common(50)
        if count >= 3 and not all(w in stopwords for w in phrase.split())
    ][:15]

    # ── 3. EMOTIONAL TONE ──────────────────────────────────────
    emotion_keywords = {
        'love':     ['love','pyaar','ishq','❤️','💕','😍','heart'],
        'humor':    ['haha','hehe','lol','😂','🤣','funny','mast','pagal'],
        'worry':    ['tension','worry','please','please','careful','dhyan'],
        'advice':   ['should','must','always','never','remember','beta',
                     'suno','sun','dekho'],
        'pride':    ['proud','shabash','well done','great','excellent',
                     'bahut accha','wah'],
        'spiritual':['god','bhagwan','allah','waheguru','bless','dua',
                     'prayer','mandir']
    }

    emotion_scores = {}
    lower_text = full_text.lower()
    for emotion, keywords in emotion_keywords.items():
        score = sum(lower_text.count(kw) for kw in keywords)
        emotion_scores[emotion] = score

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    personality['emotion_scores'] = emotion_scores
    personality['dominant_emotion'] = dominant_emotion

    # ── 4. COMMUNICATION STYLE ─────────────────────────────────
    avg_msg_length = sum(len(m.split()) for m in messages) / len(messages)
    question_count = sum(1 for m in messages if '?' in m)
    exclamation_count = sum(1 for m in messages if '!' in m)
    hindi_indicators = sum(1 for m in messages
                           if any(w in m.lower() for w in
                                  ['hai','hain','kya','nahi','aur','tum',
                                   'aap','mein','hum','yaar','beta','bhai']))

    personality['style'] = {
        'avg_message_length': round(avg_msg_length, 1),
        'asks_questions': question_count > len(messages) * 0.1,
        'expressive': exclamation_count > len(messages) * 0.15,
        'uses_hindi': hindi_indicators > len(messages) * 0.2,
        'total_messages': len(messages)
    }

    # ── 5. SIGNATURE PHRASES (exact repeated sentences) ────────
    full_sentences = [m.strip() for m in messages if len(m.split()) >= 4]
    sentence_counter = Counter(full_sentences)
    personality['signature_phrases'] = [
        phrase for phrase, count in sentence_counter.most_common(20)
        if count >= 2
    ][:10]

    # ── 6. PERSONALITY SUMMARY ─────────────────────────────────
    summary_parts = []

    if personality['style']['uses_hindi']:
        summary_parts.append("communicates in Hinglish (Hindi + English mix)")
    if personality['style']['asks_questions']:
        summary_parts.append("asks caring questions frequently")
    if personality['style']['expressive']:
        summary_parts.append("very expressive and enthusiastic")
    if avg_msg_length < 5:
        summary_parts.append("sends short, punchy messages")
    elif avg_msg_length > 15:
        summary_parts.append("writes long, detailed messages")
    if dominant_emotion == 'advice':
        summary_parts.append("loves giving guidance and advice")
    elif dominant_emotion == 'love':
        summary_parts.append("deeply affectionate and caring")
    elif dominant_emotion == 'humor':
        summary_parts.append("has a playful, funny personality")

    personality['summary'] = f"{target_name} " + ", ".join(summary_parts)
    personality['name'] = target_name

    return personality


def print_personality_report(personality: dict):
    """Prints a beautiful personality report"""

    name = personality['name']
    print("\n" + "="*55)
    print(f"  🧠 SMRITI — PERSONALITY PROFILE: {name.upper()}")
    print("="*55)

    print(f"\n📝 SUMMARY:\n   {personality['summary']}")

    print(f"\n💬 TOP WORDS THEY USED:")
    top5 = personality['top_words'][:5]
    for word, count in top5:
        print(f"   '{word}' — {count} times")

    print(f"\n🗣️ CATCHPHRASES:")
    for phrase in personality['catchphrases'][:5]:
        print(f"   • \"{phrase}\"")

    print(f"\n❤️ DOMINANT EMOTION: {personality['dominant_emotion'].upper()}")
    print(f"   Emotion Breakdown:")
    for emotion, score in personality['emotion_scores'].items():
        if score > 0:
            bar = '█' * min(score, 30)
            print(f"   {emotion:<12} {bar} ({score})")

    print(f"\n✍️ SIGNATURE PHRASES:")
    for phrase in personality['signature_phrases'][:3]:
        print(f"   • \"{phrase}\"")

    style = personality['style']
    print(f"\n📊 COMMUNICATION STYLE:")
    print(f"   Average message length : {style['avg_message_length']} words")
    print(f"   Total messages parsed  : {style['total_messages']}")
    print(f"   Uses Hindi/Hinglish    : {style['uses_hindi']}")
    print(f"   Asks questions often   : {style['asks_questions']}")
    print(f"   Very expressive        : {style['expressive']}")
    print("\n" + "="*55)
    print("  ✅ Personality profile ready for Smriti AI engine")
    print("="*55 + "\n")