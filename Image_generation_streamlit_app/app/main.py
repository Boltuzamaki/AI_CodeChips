import os

import streamlit as st
from auto1111sdk import StableDiffusionPipeline
from PIL import Image

st.set_page_config(page_title="Stable Diffusion Image Generator")


@st.cache_resource
def load_model(model_path):
    return StableDiffusionPipeline(model_path)


st.title("Stable Diffusion Image Generator")

# Specify the models directory
models_dir = "./models/"

# Get a list of all files in the models directory
model_files = [
    f
    for f in os.listdir(models_dir)
    if os.path.isfile(os.path.join(models_dir, f))
]

if model_files:
    # Display the models in a dropdown
    selected_model = st.selectbox("Select a model to use", model_files)
    model_path = os.path.join(models_dir, selected_model)

    # Load the selected model
    pipe = load_model(model_path)

    positive_prompt = st.text_area(
        "Positive Prompt",
        value=(
            "Create a highly detailed, realistic portrait of an"
            " ancient, beautiful lady. She has an ethereal grace with"
            " long, flowing dark hair adorned with intricate braids "
            "and small gold ornaments. Her skin is smooth and "
            "fair, with a soft glow as if illuminated by gentle "
            "sunlight. She wears a traditional, flowing robe made of "
            "fine, richly-colored fabric, decorated with elaborate "
            "patterns and embroidery. Her eyes are deep and "
            "expressive, reflecting wisdom and serenity, with a hint "
            "of mystery. The background should feature an ancient, "
            "majestic landscape with rolling hills, tall trees, and "
            "a serene sky. The overall atmosphere should evoke a "
            "sense of timeless beauty and elegance. 8k, realistic"
        ),  # noqa
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value=(
            "lowres, text, error, cropped, worst quality, "
            "low quality, jpeg artifacts, ugly, duplicate, "
            "morbid, mutilated, out of frame, extra fingers, "
            "mutated hands, poorly drawn hands, poorly drawn "
            "face, mutation, deformed, blurry, dehydrated, bad "
            "anatomy, bad proportions, extra limbs, cloned face, "
            "disfigured, gross proportions, malformed limbs, "
            "missing arms, missing legs, extra arms, extra legs, "
            "fused fingers, too many fingers, long neck, "
            "username, watermark, signature"
        ),
    )

    sampler_name = st.selectbox(
        "Select Sampler",
        [
            "Euler a",
            "Euler",
            "LMS",
            "Heun",
            "DPM2",
            "DPM2 a",
            "DPM++ 2S a",
            "DPM++ 2M",
            "DPM fast",
            "DPM adaptive",
            "LMS Karras",
            "DPM2 Karras",
            "DPM2 a Karras",
            "DPM++ 2S a Karras",
            "DPM++ 2M Karras",
            "DDIM",
            "PLMS",
        ],
    )

    seed = st.number_input(
        "Seed (-1 for random)",
        min_value=-1,
        value=-1,
        step=1,
    )
    steps = st.slider(
        "Steps",
        min_value=1,
        max_value=100,
        value=20,
    )
    height = st.number_input(
        "Height",
        min_value=256,
        max_value=2048,
        value=512,
        step=16,
    )
    width = st.number_input(
        "Width",
        min_value=256,
        max_value=2048,
        value=512,
        step=16,
    )
    cfg_scale = st.slider(
        "CFG Scale",
        min_value=1.0,
        max_value=30.0,
        value=7.5,
        step=0.1,
    )

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                output = pipe.generate_txt2img(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    sampler_name=sampler_name,
                    seed=seed,
                    steps=steps,
                    height=height,
                    width=width,
                    cfg_scale=cfg_scale,
                )

                st.image(
                    output[0],
                    caption="Generated Image",
                    use_column_width=True,
                )

                st.success("Image generated successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("No model files found in the specified directory.")
