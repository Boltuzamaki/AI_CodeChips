from dotenv import load_dotenv
import os 
import streamlit as st 
from app.src.youtube_audio_loader import youtube_transcriber
from app.src.summarizer import DocumentSummarizer
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Youtube Video Summarizer")

if 'youtube_video_link' not in st.session_state:
    st.session_state.youtube_video_link = ""
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'summarized' not in st.session_state:
    st.session_state.summarized = False
if 'summarizing' not in st.session_state:
    st.session_state.summarizing = False

st.sidebar.title("Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"])

use_whisper_api = st.sidebar.checkbox("Use Whisper API for Transcribe", value=False)
if use_whisper_api:
    st.sidebar.warning("Using the Whisper API may incur costs.")

transcribe_button = st.sidebar.button("Transcribe Video")
summarize_button = st.sidebar.button("Summarize Transcription")
clear_button = st.sidebar.button("Clear and Try New Video")

st.title("YouTube Video Transcriber & Summarizer")

youtube_video_link_input = st.text_input("Enter YouTube video link:", value=st.session_state.youtube_video_link)

if youtube_video_link_input != st.session_state.youtube_video_link:
    st.session_state.youtube_video_link = youtube_video_link_input
    st.session_state.docs = None
    st.session_state.transcription = None
    st.session_state.summary = None
    st.session_state.summarized = False

if st.session_state.youtube_video_link:
    st.video(st.session_state.youtube_video_link)

transcription_container = st.empty()
summary_container = st.empty()

if transcribe_button:
    with st.spinner("Transcribing..."):
        st.session_state.docs = youtube_transcriber(st.session_state.youtube_video_link, local=not use_whisper_api)
        st.session_state.transcription = "\n".join([doc.page_content for doc in st.session_state.docs])
        st.success("Transcription completed!")

if st.session_state.transcription:
    transcription_container.text_area("Transcription", value=st.session_state.transcription, height=300)

if summarize_button:
    st.session_state.summarizing = True

if st.session_state.summarizing:
    summary_container.empty()
    with st.spinner("Summarizing..."):
        llm = ChatOpenAI(api_key=openai_key, model_name=model_name)
        summarizer = DocumentSummarizer(llm=llm)
        st.session_state.summary = summarizer.summarize_documents(st.session_state.docs)
        st.session_state.summarized = True
        st.session_state.summarizing = False
        st.success("Summarization completed!")

if st.session_state.summarized:
    summary_container.text_area("Summary", value=st.session_state.summary, height=200)

if clear_button:
    st.session_state.youtube_video_link = ""
    st.session_state.docs = None
    st.session_state.transcription = None
    st.session_state.summary = None
    st.session_state.summarized = False
    st.session_state.summarizing = False
    st.rerun()
