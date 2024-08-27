import base64
import json
import os
from mimetypes import guess_type

import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()
st.set_page_config(page_title="Invoice Data Extractor")


def local_image_to_data_url(image_path):
    """
    Converts a local image file to a base64-encoded data URL.

    Args:
        image_path (str): The path to the local image file.

    Returns:
        str: A base64-encoded data URL representation of the image.
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode(
            "utf-8"
        )
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_image_prompt(image_path):
    """
    Generates a prompt template for extracting information from an image.

    Args:
        image_path (str): The path to the local image file.

    Returns:
        ChatPromptTemplate: A prompt template ready to be invoked by the LLM.
    """
    encoded_image_url = local_image_to_data_url(image_path)
    prompt_template = HumanMessagePromptTemplate.from_template(
        template=[
            {
                "type": "text",
                "text": "Extract all information from image and return only JSON format. {format_instructions}",
            },
            {
                "type": "image_url",
                "image_url": encoded_image_url,
            },
        ],
    )
    image_prompt = ChatPromptTemplate.from_messages([prompt_template])
    image_prompt.partial(
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )
    return image_prompt


def process_images(uploaded_files, llm):
    """
    Processes uploaded images, extracts information using the LLM, and displays results.

    Args:
        uploaded_files (list): List of uploaded image files.
        llm (ChatOpenAI): The LLM model instance to be used for processing.
    """
    results = {}
    for uploaded_file in uploaded_files:
        temp_image_path = save_temp_image(uploaded_file)
        json_response = extract_information(temp_image_path, llm)
        results[uploaded_file.name] = json_response

        display_results(temp_image_path, uploaded_file.name, json_response)
        cleanup_temp_file(temp_image_path)

    return results


def save_temp_image(uploaded_file):
    """
    Saves the uploaded image to a temporary file.

    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit.

    Returns:
        str: Path to the saved temporary image.
    """
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_image_path


def extract_information(image_path, llm):
    """
    Extracts information from an image using the LLM.

    Args:
        image_path (str): Path to the image file.
        llm (ChatOpenAI): The LLM model instance to be used for processing.

    Returns:
        dict: JSON response containing the extracted information.
    """
    prompt = generate_image_prompt(image_path)
    chain = prompt | llm | parser
    json_response = chain.invoke(input=[])
    return json_response


def display_results(image_path, image_name, json_response):
    """
    Displays the image and its corresponding JSON response in the Streamlit app.

    Args:
        image_path (str): Path to the image file.
        image_name (str): Name of the image file.
        json_response (dict): JSON response containing the extracted information.
    """
    st.image(image_path, caption=image_name)


def cleanup_temp_file(file_path):
    """
    Removes the temporary file created during the process.

    Args:
        file_path (str): Path to the temporary file.
    """
    os.remove(file_path)


def main():
    """
    The main function that runs the Streamlit app.
    """
    st.title("Invoice Data Extractor")

    # Persistent states for API key and model selection
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = "gpt-4o"
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = None
    if "extracted_results" not in st.session_state:
        st.session_state["extracted_results"] = None

    # User inputs for API key and model selection
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

    # File uploader remains visible even after files are uploaded
    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files

    # Show the extract button only if files have been uploaded
    if st.session_state["uploaded_files"]:
        if st.button("Extract"):
            # Initialize the LLM with the selected model and API key
            llm = ChatOpenAI(
                model=st.session_state["model_name"],
                openai_api_key=st.session_state["api_key"],
                temperature=0,
            )

            st.session_state["extracted_results"] = process_images(
                st.session_state["uploaded_files"], llm
            )

    # Display the extracted results if available
    if st.session_state["extracted_results"]:
        for image_name, json_response in st.session_state[
            "extracted_results"
        ].items():
            st.subheader(f"Results for {image_name}")
            st.json(json_response)

        if st.button("Download JSON results"):
            json_output = json.dumps(
                st.session_state["extracted_results"],
                indent=4,
                ensure_ascii=False,
            )
            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="extracted_data.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
