import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Image Caption Generator", page_icon="✨", layout="centered")

# ==========================================
# 1. Loading SOTA Model (BLIP)
# ==========================================
@st.cache_resource
def load_models():
    # Detect Apple Silicon (M1/M2/M3) for hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the processor and model from Hugging Face
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    return processor, model, device

def generate_caption(image, processor, model, device):
    # Process the image and move it to the M1 GPU
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
        
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ==========================================
# 2. Streamlit UI & Custom CSS
# ==========================================
st.markdown("""
    <style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #95a5a6;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 5px;
    }
    .caption-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .caption-box {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px;
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: #2c3e50;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        max-width: 80%;
    }
    .stButton>button {
        background-color: #4ECDC4;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 10px 24px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45b7d1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AI Image Captioner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Salesforce BLIP (State-of-the-Art Vision-Language Model)</p>', unsafe_allow_html=True)

try:
    with st.spinner("Loading Foundation Model (this may take a few seconds on first run)..."):
        processor, model, device = load_models()
except Exception as e:
    st.error(f"Error loading models. Details: {e}")
    st.stop()

st.markdown("<hr style='border:1px dashed #e0e0e0; margin-bottom:2rem;'/>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use columns to center the image effectively
            st.image(image, use_container_width=True, caption="Uploaded Image")
        
        generate_btn = st.button("✨ Generate Caption", use_container_width=True)
        
        if generate_btn:
            with st.spinner("Analyzing image and generating caption..."):
                caption = generate_caption(image, processor, model, device)
                
                # Format output nicely
                formatted_caption = caption.capitalize()
                if not formatted_caption.endswith("."):
                    formatted_caption += "."
                    
                st.markdown(f'''
                    <div class="caption-container">
                        <div class="caption-box">
                            "{formatted_caption}"
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.balloons()
    except Exception as e:
        st.error(f"An error occurred: {e}")