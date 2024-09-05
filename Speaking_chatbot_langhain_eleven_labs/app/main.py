import base64
from datetime import datetime
import speech_recognition as sr
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import requests

CHUNK_SIZE = 1024

st.set_page_config(page_title="Speaking Conversational Chatbot")

def generate_tts_audio(text, api_key, voice_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    audio_file = f"output.mp3"
    with open(audio_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    return audio_file

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] {
        position: fixed;
        bottom: 10px;
        left: calc(35% + 20px); 
        right: 10px;
        padding: 10px;
        z-index: 9000;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Configuration")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model = st.sidebar.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"], index=0)

elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
voice_id = st.sidebar.text_input("Voice ID", value="cgSgspJ2msm6clMCkdW9")

if openai_api_key:
    llm = ChatOpenAI(openai_api_key=openai_api_key, model=model)
else:
    st.error("Please enter your OpenAI API key.")

if "store" not in st.session_state:
    st.session_state.store = {}
if "session_ids" not in st.session_state:
    st.session_state.session_ids = ["1"]
if "current_session" not in st.session_state:
    st.session_state.current_session = "1"
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "new_prompt" not in st.session_state:
    st.session_state.new_prompt = None  

st.sidebar.title("Session Management")
new_session_id = st.sidebar.text_input("Add New Session")

if st.sidebar.button("Add Session"):
    if new_session_id and new_session_id not in st.session_state.session_ids:
        st.session_state.session_ids.append(new_session_id)
        st.session_state.current_session = new_session_id

session_id = st.sidebar.radio(
    "Select Session",
    st.session_state.session_ids,
    index=st.session_state.session_ids.index(st.session_state.current_session),
)
st.session_state.current_session = session_id

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
        return st.session_state.store[session_id]

    memory = ConversationBufferWindowMemory(
        chat_memory=st.session_state.store[session_id],
        k=3,
        return_messages=True,
    )
    key = memory.memory_variables[0]
    messages = memory.load_memory_variables({})[key]
    st.session_state.store[session_id] = InMemoryChatMessageHistory(
        messages=messages
    )
    return st.session_state.store[session_id]

def display_message(role, content, timestamp):
    with st.chat_message(role):
        st.markdown(f"**{role.capitalize()}:** {content}")
        st.markdown(f"<small><i>{timestamp}</i></small>", unsafe_allow_html=True)

if openai_api_key:
    chain = RunnableWithMessageHistory(llm, get_session_history)

    st.title("Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if session_id in st.session_state.messages:
        for message in st.session_state.messages[session_id]:
            display_message(message["role"], message["content"], message["timestamp"])

    col1, col2 = st.columns([4, 1])

    with col1:
        prompt = st.chat_input("What is up?")
    
    with col2:
        listen_clicked = st.button("Listen")

    if listen_clicked:
        with st.spinner("Listening..."):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    prompt = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    st.write("Google Speech Recognition could not understand audio.")
                    prompt = None
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition service: {e}")
                    prompt = None

    if prompt:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        display_message("user", prompt, timestamp)

        if session_id not in st.session_state.messages:
            st.session_state.messages[session_id] = []
        st.session_state.messages[session_id].append(
            {"role": "user", "content": prompt, "timestamp": timestamp}
        )

        with st.spinner("Thinking..."):
            response = chain.invoke(
                prompt,
                config={"configurable": {"session_id": session_id}},
            )
            bot_response = response.content

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        display_message("bot", bot_response, timestamp)
        
        st.session_state.messages[session_id].append(
            {"role": "bot", "content": bot_response, "timestamp": timestamp}
        )

        if elevenlabs_api_key and voice_id:
            audio_file = generate_tts_audio(bot_response, elevenlabs_api_key, voice_id)
            autoplay_audio(audio_file)
