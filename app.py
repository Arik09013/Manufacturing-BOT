import streamlit as st
from llm_core import ask_bot

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Manufacturing AI Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# ChatGPT-like CSS
# ----------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Global Reset ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #212121 !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a;
}

/* ── Main chat area ── */
.main-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 760px;
    margin: 0 auto;
    padding: 0 16px;
}

/* ── Top title bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 18px 0 10px;
    gap: 10px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 8px;
}
.top-bar h1 {
    font-size: 17px;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.2px;
}
.top-bar .icon {
    font-size: 20px;
}

/* ── Welcome screen ── */
.welcome-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    padding: 60px 20px;
    text-align: center;
    gap: 12px;
}
.welcome-screen .big-icon {
    font-size: 48px;
    margin-bottom: 8px;
}
.welcome-screen h2 {
    font-size: 26px;
    font-weight: 600;
    color: #ececec;
}
.welcome-screen p {
    font-size: 14px;
    color: #8e8e8e;
    max-width: 420px;
    line-height: 1.6;
}

/* ── Suggestion chips ── */
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-top: 24px;
}
.chip {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px;
    color: #c5c5c5;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
    max-width: 200px;
    text-align: center;
    line-height: 1.4;
}
.chip:hover {
    background: #333;
    border-color: #555;
    color: #fff;
}

/* ── Chat messages ── */
.chat-scroll-area {
    flex: 1;
    overflow-y: auto;
    padding: 12px 0;
    scrollbar-width: thin;
    scrollbar-color: #3a3a3a transparent;
}

/* ── Individual message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 6px 0 !important;
    border: none !important;
    max-width: 760px;
}

/* User messages - right-aligned bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div[class*="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    display: flex;
    justify-content: flex-end;
}

[data-testid="chatAvatarIcon-user"] ~ div {
    background: #2f6fed !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    max-width: 75%;
    color: #fff !important;
    font-size: 14.5px;
    line-height: 1.6;
}

/* Assistant messages - left-aligned */
[data-testid="chatAvatarIcon-assistant"] ~ div {
    background: #2a2a2a !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    max-width: 85%;
    color: #ececec !important;
    font-size: 14.5px;
    line-height: 1.7;
    border: 1px solid #333;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: #2f6fed !important;
    color: white !important;
    border-radius: 50% !important;
    font-size: 13px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #1a6b3c !important;
    color: white !important;
    border-radius: 50% !important;
    font-size: 13px !important;
}

/* ── Input area ── */
.stChatInputContainer, div[data-testid="stChatInput"] {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 16px !important;
    padding: 4px 4px !important;
    margin: 8px 0 12px !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #666 !important;
}

/* Send button */
div[data-testid="stChatInput"] button {
    background: #2f6fed !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 6px 10px !important;
}
div[data-testid="stChatInput"] button:hover {
    background: #1a56d4 !important;
}

/* ── Typing indicator (spinner) ── */
.stSpinner > div {
    border-top-color: #2f6fed !important;
}

/* ── Code blocks inside messages ── */
code {
    background: #1a1a1a !important;
    color: #e06c75 !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 13px;
}
pre code {
    display: block;
    padding: 12px !important;
    overflow-x: auto;
    font-size: 13px !important;
}

/* ── Markdown inside assistant ── */
[data-testid="chatAvatarIcon-assistant"] ~ div p { margin-bottom: 6px; }
[data-testid="chatAvatarIcon-assistant"] ~ div ul,
[data-testid="chatAvatarIcon-assistant"] ~ div ol {
    margin-left: 18px;
    margin-bottom: 6px;
}
[data-testid="chatAvatarIcon-assistant"] ~ div li { margin-bottom: 3px; }

/* ── Italic grammar correction note ── */
[data-testid="chatAvatarIcon-assistant"] ~ div em {
    color: #888;
    font-size: 12px;
}

/* ── Footer disclaimer ── */
.footer-note {
    text-align: center;
    font-size: 11.5px;
    color: #555;
    padding: 6px 0 14px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session state init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggestion_clicked" not in st.session_state:
    st.session_state.suggestion_clicked = None

# ----------------------------
# Top bar
# ----------------------------
st.markdown("""
<div class="top-bar">
    <span class="icon">🏭</span>
    <h1>Manufacturing AI Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Suggestion chips (shown only when no chat history)
# ----------------------------
SUGGESTIONS = [
    "⚙️ What is CNC machining?",
    "🔩 Explain tolerance in manufacturing",
    "🔥 Types of welding processes",
    "🏗️ What is metal casting?",
    "📐 How does a lathe work?",
    "🔬 Quality control in manufacturing",
]

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-screen">
        <div class="big-icon">🏭</div>
        <h2>Manufacturing AI Assistant</h2>
        <p>Ask me anything about manufacturing, machining, CNC, welding, casting, industrial processes, and engineering measurements.</p>
    </div>
    """, unsafe_allow_html=True)

    # Render suggestion buttons using Streamlit columns
    cols = st.columns(3)
    for idx, suggestion in enumerate(SUGGESTIONS):
        with cols[idx % 3]:
            if st.button(suggestion, key=f"chip_{idx}", use_container_width=True):
                st.session_state.suggestion_clicked = suggestion
                st.rerun()

# ----------------------------
# Handle suggestion click
# ----------------------------
if st.session_state.suggestion_clicked:
    user_input = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🏭"):
        with st.spinner("Thinking..."):
            response = ask_bot(user_input, history=st.session_state.messages)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------------------
# Display chat history
# ----------------------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏭"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Message Manufacturing Assistant...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🏭"):
        with st.spinner("Thinking..."):
            response = ask_bot(user_input, history=st.session_state.messages)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    '<div class="footer-note">Specialized in manufacturing & industrial topics only · Responds in your language</div>',
    unsafe_allow_html=True
)
