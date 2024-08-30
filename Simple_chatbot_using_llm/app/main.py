from datetime import datetime

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Conversational Chatbot")
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

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    def display_message(role, content, timestamp):
        with st.chat_message(role):
            st.markdown(f"**{role.capitalize()}:** {content}")
            st.markdown(
                f"<small><i>{timestamp}</i></small>", unsafe_allow_html=True
            )

    if session_id in st.session_state.messages:
        for message in st.session_state.messages[session_id]:
            display_message(
                message["role"], message["content"], message["timestamp"]
            )

    if prompt := st.chat_input("What is up?"):
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
