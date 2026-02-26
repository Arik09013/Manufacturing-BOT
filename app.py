import streamlit as st
from llm_core import ask_bot

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="Manufacturing AI Assistant",
    layout="wide"
)
st.title("🤖 Manufacturing AI Assistant")

# ----------------------------
# Initialize session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display chat history
# ----------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.markdown(content)

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask me anything about manufacturing...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    with st.chat_message("assistant"):
        # Pass full conversation history if you want context-aware answers
        response = ask_bot(user_input, history=st.session_state.messages)
        st.markdown(response)

    # Save assistant response in session
    st.session_state.messages.append({"role": "assistant", "content": response})
