from datetime import datetime

import speech_recognition as sr
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Conversational Chatbot")

st.markdown(
    """
    <style>
    /* Adjust the horizontal block */
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
model = st.sidebar.selectbox(
    "Model", options=["gpt-4o", "gpt-4o-mini"], index=0
)

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
    st.session_state.new_prompt = None  # Temporary storage for the new prompt

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


if openai_api_key:
    chain = RunnableWithMessageHistory(llm, get_session_history)

    st.title("Chatbot")

    # Display chat messages
    if session_id in st.session_state.messages:
        for message in st.session_state.messages[session_id]:
            with st.chat_message(message["role"]):
                st.markdown(
                    f"**{message['role'].capitalize()}: {message['content']}**"
                )
                st.markdown(
                    f"<small><i>{message['timestamp']}</i></small>",
                    unsafe_allow_html=True,
                )

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
                    st.session_state.new_prompt = recognizer.recognize_google(
                        audio
                    )
                except sr.UnknownValueError:
                    st.write(
                        "Google Speech Recognition could not understand audio"
                    )
                    st.session_state.new_prompt = None
                except sr.RequestError as e:
                    st.write(
                        (
                            "Could not request results from Google "
                            f"Speech Recognition service; {e}"
                        )
                    )
                    st.session_state.new_prompt = None

    if prompt or st.session_state.new_prompt:
        active_prompt = prompt or st.session_state.new_prompt
        st.session_state.new_prompt = None  # Reset after processing

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.setdefault(session_id, []).append(
            {"role": "user", "content": active_prompt, "timestamp": timestamp}
        )

        with st.spinner("Thinking..."):
            response = chain.invoke(
                active_prompt,
                config={"configurable": {"session_id": session_id}},
            )
            bot_response = response.content

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages[session_id].append(
            {
                "role": "bot",
                "content": bot_response,
                "timestamp": timestamp,
            }
        )
        st.rerun()
