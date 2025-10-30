import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────
# Page config + Dark Theme
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    body, .stApp { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3, h4, h5, h6 { color: #fafafa; }
    .stMarkdown p, .stMarkdown span, .stTextInput>div>div>input { color: #d1d5db !important; }
    .stButton>button { color: white; background-color: #262730; border: 1px solid #3c4048; border-radius: 8px; padding: 0.5em 1.2em; }
    .stButton>button:hover { background-color: #3c4048; color: #fff; }
    .stFileUploader>div>div>button { background-color: #262730; color: #fafafa; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 15px; background: linear-gradient(90deg, #0e1117, #1a1d23); border-radius: 10px;'>
<h1 style='color: #00b4d8;'>🤖 Digit Recognition AI</h1>
<p style='color: #9ca3af;'>Draw or upload a handwritten digit (0–9). The model will predict the digit and show confidence.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model Loading
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
# Input area (upload + draw)
# ──────────────────────────────────────────────
st.markdown("### 🖌️ Upload or Draw a Digit")

left, right = st.columns([1,1])

with left:
    uploaded_file = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"])
    st.markdown("**Or draw below (white stroke on black background recommended):**")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=12,
        stroke_color="#ffffff",
        background_color="#000000",
        height=260,
        width=260,
        drawing_mode="freedraw",
        key="canvas"
    )
    predict_btn = st.button("🔍 Recognize Digit")

with right:
    st.subheader("Preview")
    # preview area will be updated after prediction
    preview_placeholder = st.empty()

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def pil_from_canvas(canvas_obj):
    """
    Convert canvas.image_data (numpy RGBA) -> PIL.Image (grayscale)
    Returns None if no drawing.
    """
    if canvas_obj is None:
        return None
    arr = canvas_obj.image_data
    if arr is None:
        return None
    try:
        img = Image.fromarray(arr.astype('uint8'), 'RGBA').convert('L')
        # Crop to bounding box of drawing to remove empty margins (optional)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        return img
    except Exception:
        return None

def preprocess_pil(pil_img):
    """Make image MNIST-like: invert, resize to 28x28, normalize"""
    img = pil_img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr, img

# ──────────────────────────────────────────────
# Prediction flow
# ──────────────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("Model not loaded. Please add model/digit_recognition_model.keras and restart.")
    else:
        pil_img = None
        # Priority: uploaded image > canvas
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert('L')
            except Exception as e:
                st.error(f"Failed to open uploaded image: {e}")
        else:
            pil_img = pil_from_canvas(canvas_result)

        if pil_img is None:
            st.warning("No image found. Upload a PNG/JPG or draw on the canvas, then click Recognize.")
        else:
            arr, preview_img = preprocess_pil(pil_img)
            preds = model.predict(arr)
            digit = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0])) * 100.0

            # Show results
            preview_placeholder.image(preview_img.resize((140,140)), caption="Processed (28x28)", width=140)
            st.success(f"🎯 Predicted Digit: **{digit}**")
            st.info(f"Confidence: **{conf:.2f}%**")

            probs = {str(i): float(preds[0][i]) for i in range(10)}
            st.bar_chart([probs])

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<hr style='border: 1px solid #3c4048;'>
<div style='text-align:center; color:#6b7280;'>
<p>Made with ❤️ by <b>Abu Sufyan – Student</b> | Organization: <b>Abu Zar</b></p>
</div>
""", unsafe_allow_html=True)# 📚 FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<hr style='border: 1px solid #3c4048;'>
<div style='text-align:center; color:#6b7280;'>
<p>Made with ❤️ by <b>Abu Sufyan – Student</b> | Organization: <b>Abu Zar</b></p>
</div>
""", unsafe_allow_html=True)
