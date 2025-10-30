import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
from model_utils import load_model_or_prompt, preprocess_pil_image
from streamlit_drawable_canvas import st_canvas

# Page config & styling
st.set_page_config(page_title="Digit Recognition AI", page_icon="🤖", layout="centered")
st.markdown("""<style>
.stApp { background: linear-gradient(180deg,#f8fafc 0%, #eef2ff 100%); }
.title { font-size:32px; font-weight:700; color:#0f172a; }
.subtitle { color:#334155; }
.card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title">Digit Recognition AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive chat-style interface — draw or upload a handwritten digit and the model will respond.</div>', unsafe_allow_html=True)
st.write('')

MODEL_PATH = os.path.join('model','digit_recognition_model.keras')
model = load_model_or_prompt(MODEL_PATH)

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role":"bot","text":"Hello — draw or upload a digit (0–9). I will guess it for you."}]

def add_user(msg):
    st.session_state.messages.append({"role":"user","text":msg})
def add_bot(msg):
    st.session_state.messages.append({"role":"bot","text":msg})

left, right = st.columns([1,1])
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Input')
    uploaded = st.file_uploader('Upload an image (png/jpg)', type=['png','jpg','jpeg'])
    st.markdown('**Or draw below (white ink on black background recommended):**')
    canvas_result = st_canvas(
        fill_color='rgba(255, 255, 255, 1)',
        stroke_width=12,
        stroke_color='#ffffff',
        background_color='#000000',
        height=260,
        width=260,
        drawing_mode='freedraw',
        key='canvas'
    )
    predict = st.button('Predict')
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Conversation')
    for m in st.session_state.messages:
        if m['role']=='bot':
            st.markdown(f"**🤖 Bot:** {m['text']}")
        else:
            st.markdown(f"**You:** {m['text']}")
    st.markdown('</div>', unsafe_allow_html=True)

def image_from_canvas(canvas_obj):
    if canvas_obj is None:
        return None
    arr = canvas_obj.image_data
    if arr is None:
        return None
    from PIL import Image
    img = Image.fromarray(arr.astype('uint8'), 'RGBA').convert('L')
    return img

if predict:
    pil = None
    if uploaded:
        pil = Image.open(uploaded).convert('L')
        add_user('Uploaded an image.')
    else:
        pil = image_from_canvas(canvas_result)
        if pil is not None:
            add_user('Drew a digit on the canvas.')

    if pil is None:
        add_bot('No image found. Please upload a PNG/JPG or draw on the canvas.')
    else:
        arr, preview = preprocess_pil_image(pil)
        preds = model.predict(arr)
        digit = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))*100.0
        add_bot(f'I think the digit is **{digit}** with **{conf:.2f}%** confidence.')
        st.image(preview.resize((140,140)), caption='Processed (28x28)', width=140)
        probs = {str(i): float(preds[0][i]) for i in range(10)}
        st.bar_chart([probs])
