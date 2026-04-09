import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import math
import nltk
from collections import Counter
import os

# Set page configuration
st.set_page_config(page_title="Image Caption Generator", page_icon="✨", layout="centered")

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# ==========================================
# 1. Model & Vocabulary Classes Definition
# ==========================================

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx = 4

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized_text]

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = self.conv(features)
        features = self.bn(features)
        features = features.flatten(2).permute(0, 2, 1)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=3, dropout=0.3):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, features, captions):
        captions_input = captions[:, :-1]
        seq_length = captions_input.size(1)
        
        embeddings = self.dropout(self.pos_encoder(self.embed(captions_input)))
        tgt_mask = self.generate_square_subsequent_mask(seq_length).to(features.device)
        tgt_key_padding_mask = (captions_input == 0)
        
        outputs = self.transformer_decoder(
            tgt=embeddings,
            memory=features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        outputs = self.linear(outputs)
        return outputs

# ==========================================
# 2. Loading Models & Helper Functions
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, 'transformer_vocab.pkl')
    encoder_path = os.path.join(base_dir, 'transformer_encoder.pth')
    decoder_path = os.path.join(base_dir, 'transformer_decoder.pth')

    # Load Vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    embed_size = 256
    vocab_size = len(vocab)
    num_heads = 8
    num_layers = 3
    decoder_dropout = 0.3
    
    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers, decoder_dropout).to(device)
    
    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, vocab

def generate_caption(image, encoder, decoder, vocab, max_length=50):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = encoder(image_tensor)
        
        caption_indices = [vocab.stoi["<SOS>"]]
        for _ in range(max_length):
            # Pad a zero so that taking :-1 gives the exact caption we have
            caption_tensor_with_pad = torch.tensor([caption_indices + [0]]).to(device)
            outputs = decoder(features, caption_tensor_with_pad)
            
            # Get the predicted ID from the last sequence element
            predicted_word_idx = outputs.argmax(2)[0, -1].item()
            caption_indices.append(predicted_word_idx)
            
            if predicted_word_idx == vocab.stoi["<EOS>"]:
                break
                
    generated_caption = [vocab.itos[idx] for idx in caption_indices if idx not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]]
    return " ".join(generated_caption)

# ==========================================
# 3. Streamlit UI
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
st.markdown('<p class="sub-title">Powered by ResNet-50 & Transformer Decoder trained on Flickr8k</p>', unsafe_allow_html=True)

try:
    with st.spinner("Loading AI Models (this may take a few seconds)..."):
        encoder, decoder, vocab = load_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure the weights and vocabulary files are present. Details: {e}")
    st.stop()

st.markdown("<hr style='border:1px dashed #e0e0e0; margin-bottom:2rem;'/>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

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
                caption = generate_caption(image, encoder, decoder, vocab)
                
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
