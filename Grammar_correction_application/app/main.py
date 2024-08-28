import difflib
from typing import List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Grammar Correction App")
parser = StrOutputParser()
template = """Give the user text: {user_input}\n\n
             Find any grammatical error and return the corrected version.
             Do not change the overall intent of the user's query, 
             just correct any grammatical mistakes. If the user query does not have any grammatical error # noqa
             just return the text as it is."""

prompt = ChatPromptTemplate.from_template(template)


def correct_text(
    input_text,
    llm,
):
    """
    Corrects the given input text using a language model.

    Args:
        input_text (str): The text to be corrected.
        llm (ChatOpenAI): The language model instance.

    Returns:
        str: The corrected text.
    """
    grammar_correction_chain = prompt | llm | parser
    return grammar_correction_chain.invoke({"user_input": input_text})


def get_incorrect_words(
    original: str,
    corrected: str,
) -> List[str]:
    """
    Identifies incorrect words in the original text by comparing
    it with the corrected text.

    Args:
        original (str): The original text.
        corrected (str): The corrected text.

    Returns:
        List[str]: A list of incorrect words that were changed in the
                   corrected text.
    """
    d = difflib.Differ()
    diff = list(d.compare(original.split(), corrected.split()))
    incorrect_words = [word[2:] for word in diff if word.startswith("- ")]
    return incorrect_words


def highlight_incorrect_words(
    input_text: str,
    incorrect_words: List[str],
) -> str:
    """
    Highlights incorrect words in the input text by underlining them
    with a wavy red line.

    Args:
        input_text (str): The original input text.
        incorrect_words (List[str]): A list of incorrect words to be
                                     highlighted.

    Returns:
        str: The text with incorrect words highlighted.
    """
    words = input_text.split()
    highlighted_text = " ".join(
        (
            f"<span style='text-decoration: underline wavy red;'>{word}</span>"
            if word in incorrect_words
            else word
        )
        for word in words
    )
    return highlighted_text


def main():
    """
    Main function to run the Streamlit app. It sets up the UI components and
    handles user interactions for grammar and spellcheck correction.
    """
    st.title("Grammar and Spellcheck Correction App")

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = "gpt-4o"

    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        value=st.session_state["api_key"],
    )
    model_name = st.selectbox(
        "Choose the model",
        ["gpt-4o", "gpt-4o-mini"],
        index=["gpt-4o", "gpt-4o-mini"].index(st.session_state["model_name"]),
    )

    st.write(
        "Enter your text below and click 'Correct' to see the corrected version with highlighted errors."  # noqa
    )
    input_text = st.text_area("Enter Text", height=200)

    if st.button("Correct"):
        st.session_state["api_key"] = api_key
        st.session_state["model_name"] = model_name

        if st.session_state["model_name"] and st.session_state["api_key"]:
            if input_text:
                llm = ChatOpenAI(
                    model=st.session_state["model_name"],
                    openai_api_key=st.session_state["api_key"],
                    temperature=0,
                )

                corrected_text = correct_text(input_text, llm)
                incorrect_words = get_incorrect_words(
                    input_text, corrected_text
                )
                highlighted_text = highlight_incorrect_words(
                    input_text, incorrect_words
                )

                st.subheader("Original Text with Highlighted Errors")
                st.markdown(highlighted_text, unsafe_allow_html=True)

                st.subheader("Corrected Text")
                st.write(corrected_text)
            else:
                st.warning("Please enter some text to correct.")
        else:
            st.warning("Please provide both the model name and API key.")


if __name__ == "__main__":
    main()
