import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ──────────────────────────────────────────────
# 🌑 PAGE CONFIG + DARK THEME
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Dark Theme Styling
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa;
    }
    .stMarkdown p, .stMarkdown span, .stTextInput>div>div>input {
        color: #d1d5db !important;
    }
    .stButton>button {
        color: white;
        background-color: #262730;
        border: 1px solid #3c4048;
        border-radius: 8px;
        padding: 0.5em 1.2em;
    }
    .stButton>button:hover {
        background-color: #3c4048;
        color: #fff;
    }
    .stFileUploader>div>div>button {
        background-color: #262730;
        color: #fafafa;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# 🧭 HEADER (Stylish Gradient)
# ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 15px; background: linear-gradient(90deg, #0e1117, #1a1d23); border-radius: 10px;'>
<h1 style='color: #00b4d8;'>🤖 Digit Recognition AI</h1>
<p style='color: #9ca3af;'>A deep learning app that identifies handwritten digits (0–9) using a trained CNN model on MNIST dataset.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🧩 MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model/digit_recognition_model.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# ──────────────────────────────────────────────
# 🖼️ UPLOAD / DRAW INTERFACE
# ──────────────────────────────────────────────
st.markdown("### 🖌️ Upload or Draw a Digit")

uploaded_file = st.file_uploader("Upload an image of a digit (0–9)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(uploaded_file, caption="Uploaded Image", width=150)
    
    if st.button("🔍 Recognize Digit"):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"🎯 Predicted Digit: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.markdown("""
        <div style='text-align:center; margin-top:20px;'>
        <p style='color:#9ca3af;'>This model is powered by <b>Convolutional Neural Networks</b> trained on <b>MNIST dataset</b>.</p>
        </div>
        """, unsafe_allow_html=True)

elif uploaded_file is None:
    st.markdown("<p style='color:#9ca3af;'>Please upload a digit image to begin prediction.</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 📚 FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<hr style='border: 1px solid #3c4048;'>
<div style='text-align:center; color:#6b7280;'>
<p>Made with ❤️ by <b>Abu Sufyan – Student</b> | Organization: <b>Abu Zar</b></p>
</div>
""", unsafe_allow_html=True)
