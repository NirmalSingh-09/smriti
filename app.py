import streamlit as st
import json
import os
import pickle
from memory.vector_store import build_memory, search_memory
from ai.responder import setup_gemini, generate_response
from parser.whatsapp_parser import parse_whatsapp_chat
from personality.extractor import extract_personality
from datetime import datetime

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Smriti 🌸",
    page_icon="🌸",
    layout="wide"
)
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)
# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(ellipse at top left, #0d0221 0%, #080010 50%, #000000 100%);
        min-height: 100vh;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .chat-header {
        background: linear-gradient(135deg,
            rgba(124,111,247,0.15) 0%,
            rgba(88,28,135,0.1) 50%,
            rgba(0,0,0,0.3) 100%);
        border-radius: 20px;
        padding: 18px 25px;
        margin-bottom: 25px;
        border: 1px solid rgba(124,111,247,0.3);
        box-shadow:
            0 0 30px rgba(124,111,247,0.15),
            0 0 60px rgba(124,111,247,0.05),
            inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .chat-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,111,247,0.8), transparent);
    }
    .user-bubble {
        background: linear-gradient(135deg, #7c6ff7 0%, #5a4fcf 50%, #4338ca 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 4px 20px;
        margin: 6px 0 6px auto;
        max-width: 65%;
        width: fit-content;
        float: right;
        clear: both;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(124,111,247,0.4);
        line-height: 1.5;
    }
    .ai-bubble {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(124,111,247,0.05) 100%);
        color: #e8e8f8;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 4px;
        margin: 6px auto 6px 0;
        max-width: 65%;
        width: fit-content;
        float: left;
        clear: both;
        font-size: 14px;
        border: 1px solid rgba(124,111,247,0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        line-height: 1.5;
    }
    .timestamp { font-size: 10px; color: rgba(124,111,247,0.5); margin-top: 2px; }
    .user-time { text-align: right; clear: both; margin-bottom: 10px; }
    .ai-time { text-align: left; clear: both; margin-bottom: 10px; }
    .clearfix::after { content: ""; display: table; clear: both; }
    .stat-card {
        background: linear-gradient(135deg, rgba(124,111,247,0.08) 0%, rgba(0,0,0,0.3) 100%);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 8px;
        border: 1px solid rgba(124,111,247,0.15);
        color: white;
    }
    .stTextInput input {
        background: rgba(124,111,247,0.05) !important;
        color: white !important;
        border: 1px solid rgba(124,111,247,0.25) !important;
        border-radius: 30px !important;
        padding: 14px 22px !important;
        font-size: 14px !important;
    }
    .stTextInput input::placeholder { color: rgba(124,111,247,0.4) !important; }
    .stButton button {
        background: linear-gradient(135deg, #7c6ff7, #5a4fcf) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 28px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(124,111,247,0.3) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080010 0%, #0d0221 100%) !important;
        border-right: 1px solid rgba(124,111,247,0.15) !important;
    }
    hr { border-color: rgba(124,111,247,0.15) !important; }
    .welcome-glow { filter: drop-shadow(0 0 30px rgba(124,111,247,0.6)); }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(124,111,247,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "ready" not in st.session_state:
    st.session_state.ready = False
if "personality" not in st.session_state:
    st.session_state.personality = None
if "collection" not in st.session_state:
    st.session_state.collection = None


@st.cache_resource
def load_model():
    return setup_gemini()

model = load_model()

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 25px 0 15px 0;'>
        <div style='font-size: 55px;
                    filter: drop-shadow(0 0 20px rgba(124,111,247,0.8));
                    margin-bottom: 12px;'>🌸</div>
        <div style='font-size: 24px; font-weight: 700;
                    background: linear-gradient(135deg, #ffffff, #7c6ff7);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;'>
            Smriti
        </div>
        <div style='font-size: 11px; color: rgba(124,111,247,0.6);
                    letter-spacing: 2px; text-transform: uppercase;
                    margin-top: 4px;'>
            AI Memory Keeper
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='color: #7c6ff7; font-weight: 600; margin-bottom: 10px;'>
        📁 Load A Memory
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload WhatsApp chat (.txt)",
        type=['txt'],
        label_visibility="collapsed"
    )

    person_name = st.text_input(
        "Person's name (as in WhatsApp)",
        placeholder="e.g. Kinder Joy",
        label_visibility="visible"
    )

    load_btn = st.button("🌸 Load Memories", use_container_width=True)

    if load_btn and uploaded_file and person_name:
        with st.spinner("Loading memories..."):
            os.makedirs("data", exist_ok=True)
            chat_path = f"data/{uploaded_file.name}"
            with open(chat_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            df = parse_whatsapp_chat(chat_path, person_name)

            if len(df) == 0:
                st.error(f"No messages found for '{person_name}'. Check the name!")
            else:
                personality = extract_personality(df, person_name)
                profile = personality.copy()
                profile['top_words'] = dict(personality['top_words'])

                with open("data/personality_profile.json", 'w', encoding='utf-8') as f:
                    json.dump(profile, f, ensure_ascii=False, indent=2)

                from sklearn.feature_extraction.text import TfidfVectorizer
                import pickle

                messages_list = df['message'].tolist()
                meaningful = [m for m in messages_list if len(m.split()) >= 3]

                vectorizer = TfidfVectorizer(max_features=5000)
                matrix = vectorizer.fit_transform(meaningful)

                os.makedirs("memory", exist_ok=True)
                with open("memory/messages.pkl", "wb") as f:
                    pickle.dump(meaningful, f)
                with open("memory/vectorizer.pkl", "wb") as f:
                    pickle.dump(vectorizer, f)
                with open("memory/matrix.pkl", "wb") as f:
                    pickle.dump(matrix, f)

                collection = {
                    "messages": meaningful,
                    "vectorizer": vectorizer,
                    "matrix": matrix
                }

                st.session_state.personality = profile
                st.session_state.collection = collection
                st.session_state.messages = []
                st.session_state.ready = True
                st.success(f"✅ {len(df)} memories loaded!")
                st.rerun()

    # Load existing if available
    if (not st.session_state.ready and
            os.path.exists("data/personality_profile.json") and
            os.path.exists("memory/messages.pkl")):
        try:
            with open("data/personality_profile.json", 'r', encoding='utf-8') as f:
                st.session_state.personality = json.load(f)
            with open("memory/messages.pkl", "rb") as f:
                messages = pickle.load(f)
            with open("memory/vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            with open("memory/matrix.pkl", "rb") as f:
                matrix = pickle.load(f)
            st.session_state.collection = {
                "messages": messages,
                "vectorizer": vectorizer,
                "matrix": matrix
            }
            st.session_state.ready = True
        except:
            pass

    # Personality stats
    if st.session_state.ready and st.session_state.personality:
        p = st.session_state.personality
        st.markdown("---")
        st.markdown("""
        <div style='color: #7c6ff7; font-weight: 600; margin-bottom: 10px;'>
            🧠 Personality Profile
        </div>
        """, unsafe_allow_html=True)

        style = p.get('style', {})
        emotion = p.get('dominant_emotion', 'unknown')
        emoji_map = {
            'humor': '😄', 'love': '❤️', 'advice': '🧠',
            'worry': '🤗', 'pride': '⭐', 'spiritual': '🙏'
        }

        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:13px; color:#8888aa;'>Dominant Mood</div>
            <div style='font-size:18px; font-weight:700;'>
                {emoji_map.get(emotion,'💬')} {emotion.title()}
            </div>
        </div>
        <div class="stat-card">
            <div style='font-size:13px; color:#8888aa;'>Messages Remembered</div>
            <div style='font-size:18px; font-weight:700; color:#7c6ff7;'>
                {style.get('total_messages', 0):,}
            </div>
        </div>
        <div class="stat-card">
            <div style='font-size:13px; color:#8888aa;'>Avg Message Length</div>
            <div style='font-size:18px; font-weight:700; color:#7c6ff7;'>
                {style.get('avg_message_length', 0)} words
            </div>
        </div>
        <div class="stat-card">
            <div style='font-size:13px; color:#8888aa;'>Language Style</div>
            <div style='font-size:18px; font-weight:700; color:#7c6ff7;'>
                {'🇮🇳 Hinglish' if style.get('uses_hindi') else '🇬🇧 English'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        top_words = list(p.get('top_words', {}).keys())[:5]
        if top_words:
            st.markdown("""
            <div style='color: #7c6ff7; font-weight: 600; margin: 10px 0 5px 0;'>
                💬 Favourite Words
            </div>
            """, unsafe_allow_html=True)
            words_html = " ".join([
                f"<span style='background:rgba(124,111,247,0.1); color:#7c6ff7;"
                f"padding:4px 10px; border-radius:20px; font-size:12px;"
                f"margin:2px; display:inline-block;"
                f"border:1px solid rgba(124,111,247,0.3);'>{w}</span>"
                for w in top_words
            ])
            st.markdown(words_html, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── MAIN CHAT AREA ───────────────────────────────────────
if not st.session_state.ready:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <div class='welcome-glow' style='font-size: 90px; margin-bottom: 25px;'>🌸</div>
        <div style='font-size: 36px; font-weight: 700;
                    background: linear-gradient(135deg, #ffffff, #7c6ff7);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 15px;'>
            Welcome to Smriti
        </div>
        <div style='font-size: 15px; color: rgba(255,255,255,0.4);
                    max-width: 380px; margin: 0 auto 40px auto; line-height: 1.8;'>
            Preserve the voice, words and personality<br>
            of someone you love — forever.
        </div>
        <div style='display:inline-block;
                    background: rgba(124,111,247,0.1);
                    border: 1px solid rgba(124,111,247,0.3);
                    border-radius: 30px; padding: 10px 25px;
                    color: rgba(124,111,247,0.8); font-size: 13px;'>
            ← Upload a WhatsApp chat to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    personality = st.session_state.personality
    collection = st.session_state.collection
    name = personality.get('name', 'Smriti')
    dominant_emotion = personality.get('dominant_emotion', 'humor')
    emoji_map = {
        'humor': '😄', 'love': '❤️', 'advice': '🧠',
        'worry': '🤗', 'pride': '⭐', 'spiritual': '🙏'
    }
    mood_emoji = emoji_map.get(dominant_emotion, '💬')

    st.markdown(f"""
    <div class="chat-header">
        <div style="font-size: 45px; margin-right: 15px; display:inline-block;">🌸</div>
        <div style="display:inline-block; vertical-align:middle;">
            <div style="font-size: 20px; font-weight: 700; color: white;">{name}</div>
            <div style="font-size: 13px; color: #7c6ff7;">{mood_emoji} Memory active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center; color: rgba(124,111,247,0.4);
                    font-size: 13px; margin: 10px 0 20px 0;">
            🌸 {name}'s memories have been preserved. Start the conversation.
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="clearfix">
                <div class="user-bubble">{msg['content']}</div>
            </div>
            <div class="user-time">
                <span class="timestamp">{msg['time']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="clearfix">
                <div class="ai-bubble">{msg['content']}</div>
            </div>
            <div class="ai-time">
                <span class="timestamp">{msg['time']} ✓</span>
            </div>
            """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "",
            placeholder=f"Message {name}...",
            key=f"input_{st.session_state.input_key}",
            label_visibility="collapsed"
        )
    with col2:
        send = st.button("Send 🚀")

    if send and user_input.strip():
        now = datetime.now().strftime("%I:%M %p")

        st.session_state.messages.append({
            'role': 'user',
            'content': user_input.strip(),
            'time': now
        })

        history = [
            f"{'You' if m['role'] == 'user' else name}: {m['content']}"
            for m in st.session_state.messages[-6:]
        ]

        memories = search_memory(user_input, collection, n_results=3)
        response = generate_response(
            user_input, memories, personality, model, history
        )

        st.session_state.messages.append({
            'role': 'ai',
            'content': response,
            'time': now
        })

        st.session_state.input_key += 1
        st.rerun()

    st.markdown("""
    <div style="text-align:center; color: rgba(124,111,247,0.2);
                font-size: 11px; margin-top: 20px;">
        🌸 Smriti — Memories preserved with love
    </div>
    """, unsafe_allow_html=True)