import torch
from googletrans import Translator
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import streamlit as st



st.title("`LITTIS` image generator")

with st.expander('About the app'):
    st.markdown("LITTIS is a mini-app that generates Images using Stable Diffusion Model based on Lingala text prompt.")

# Generate image based on a lingala prompt
@st.cache_resource
def generate_image(prompt, model_id):

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("mps")

    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    # First-time "warmup" pass (see explanation above)
    _ = pipe(prompt, num_inference_steps=1)

    # Results match those from the CPU device after the warmup pass.
    image = pipe(prompt).images[0]

    return image

# Generate okapi image based on a lingala prompt
@st.cache_resource
def generate_image_okapi(prompt, model_id, use_auth_token=None):

    pipe = StableDiffusionPipeline.from_pretrained(model_id , use_auth_token=use_auth_token)
    pipe = pipe.to("mps")

    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    # First-time "warmup" pass (see explanation above)
    _ = pipe(prompt, num_inference_steps=1)

    # Results match those from the CPU device after the warmup pass.
    image = pipe(prompt).images[0]

    return image

model_id = "CompVis/stable-diffusion-v1-4"
model_id2 = "runwayml/stable-diffusion-v1-5"
okapi_model_id = "BrainTheos/okapi-drc"
translator = Translator()

st.header('General-purpose LITTIS')

with st.expander('About the model'):
    st.write('Checkpoint used : stable-diffusion-v1-5 from Hugging Face.')

with st.expander('Try out'):
    prompt = st.text_input(label="Text", placeholder="Enter a text in Lingala")
    translation=''
    if prompt != '':
        trans = translator.translate(prompt, dest='en')
        translation = trans.text
    else:
        st.write('☝️  Please enter a text')
    print(translation)

    gen_btn = st.button(
        label="Generate Image",
        type = "primary")

    if gen_btn:
        img = generate_image(translation, model_id2)
        st.subheader('Generated Image')
        st.image(img)

# Okapi-driven model
st.header('Okapi-driven LITTIS')

with st.expander('About the model'):
    st.write('Model fine-tuned on images of Okapi mammal using DreamBooth approach')

with st.expander('Try out'):
    prompt = st.text_input(label="Prompt", placeholder="Enter a text in Lingala")
    translation=''
    if prompt != '':
        trans = translator.translate(prompt, dest='en')
        translation = trans.text
    else:
        st.write('☝️  Please enter a text')
    print(translation)

    gen_btn = st.button(
        label="Generate",
        type = "primary")

    if gen_btn:
        token = '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'
        img = generate_image_okapi(translation, okapi_model_id, token)
        st.subheader('Generated Image')
        st.image(img)


