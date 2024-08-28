import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Language Translator")

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gpt-4o"
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "target_language" not in st.session_state:
    st.session_state["target_language"] = ""

st.title("Language Translator")

st.session_state["api_key"] = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    value=st.session_state["api_key"],
)
st.session_state["model_name"] = st.selectbox(
    "Choose the model",
    ["gpt-4o", "gpt-4o-mini"],
    index=["gpt-4o", "gpt-4o-mini"].index(st.session_state["model_name"]),
)

st.session_state["input_text"] = st.text_area(
    "Enter the text you want to translate:",
    value=st.session_state["input_text"],
)
st.session_state["target_language"] = st.text_input(
    "Enter the target language (e.g., 'Japanese'):",
    value=st.session_state["target_language"],
)

if st.button("Translate"):
    llm = ChatOpenAI(
        model=st.session_state["model_name"],
        openai_api_key=st.session_state["api_key"],
        temperature=0,
    )

    template = """
    Please translate the given text into {language} language\n\n\n

    {text} \n

    Please return only the translated language
    """

    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    language_translator_chain = prompt | llm | parser

    translation = language_translator_chain.invoke(
        {
            "text": st.session_state["input_text"],
            "language": st.session_state["target_language"],
        }
    )

    st.write("Translated Text:")
    st.success(translation)
