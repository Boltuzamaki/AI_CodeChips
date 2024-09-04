import os
from datetime import datetime

import streamlit as st
from app.src.qna import ConversationalQA
from app.src.youtube_audio_loader import youtube_transcriber

if "store" not in st.session_state:
    st.session_state.store = {}
if "docs" not in st.session_state:
    st.session_state.docs = None
if "messages" not in st.session_state:
    st.session_state.messages = {}

st.set_page_config(page_title="YouTube Transcriber & Chatbot")
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

model = st.sidebar.selectbox(
    "Model", options=["gpt-4o", "gpt-4o-mini"], index=0
)

use_whisper_api = st.sidebar.checkbox(
    "Use Whisper API for Transcription", value=False
)

if use_whisper_api:
    st.sidebar.warning("Using OpenAI Whisper API may incur costs.")
    local = False
else:
    local = True

st.title("YouTube Video Transcriber & Chatbot")

youtube_link = st.text_input("Enter YouTube Video Link")

if youtube_link:
    st.video(youtube_link)

# Transcription
if st.button("Transcribe"):
    if openai_api_key:
        st.session_state.docs = youtube_transcriber(youtube_link, local=local)
        st.session_state.messages = []
        st.success("Transcription completed!")
    else:
        st.error("Please enter your OpenAI API key.")

if st.session_state.docs:
    qa_system = ConversationalQA(docs=st.session_state.docs)

    st.write("### Ask me anything!")

    def display_message(role, content, timestamp):
        with st.chat_message(role):
            st.markdown(f"**{role.capitalize()}:** {content}")
            st.markdown(
                f"<small><i>{timestamp}</i></small>", unsafe_allow_html=True
            )

    if st.session_state.messages:
        for message in st.session_state.messages:
            display_message(
                message["role"], message["content"], message["timestamp"]
            )

    if prompt := st.chat_input("Your question here..."):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        display_message("user", prompt, timestamp)

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": timestamp}
        )

        with st.spinner("Thinking..."):
            response = qa_system.invoke_chain(
                session_id="1", user_input=prompt
            )
            bot_response = response

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        display_message("bot", bot_response, timestamp)

        st.session_state.messages.append(
            {"role": "bot", "content": bot_response, "timestamp": timestamp}
        )
