import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import nltk

# Download the required NLTK tokenizers for the Streamlit Cloud server
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# 1. THE VOCABULARY CLASS (Required for Pickle to load the dictionary)
# ==========================================
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx = 4
        
    def __len__(self):
        return len(self.itos)
    
    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

# ==========================================
# 2. DEFINE THE ARCHITECTURE
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = resnet
        self.resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        return self.bn(self.resnet(images))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """Used during training only."""
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        return self.linear(self.dropout(lstm_out))

    def init_hidden_with_features(self, features):
        """
        Seed the LSTM hidden state using the image features.
        This matches how the training forward pass starts: features are the
        first token fed into the LSTM, so we run one step to get the initial
        hidden/cell state for autoregressive decoding.
        """
        # features: (1, embed_size)
        lstm_input = features.unsqueeze(1)          # (1, 1, embed_size)
        _, hidden = self.lstm(lstm_input)            # hidden = (h, c), each (1, 1, hidden_size)
        return hidden

    def decode_step(self, word_idx, hidden):
        """
        Run a single decoding step.
        Returns:
            logits  : (1, vocab_size)  — unnormalised scores for every word
            hidden  : updated (h, c) tuple to pass to the next step
        """
        embedding = self.embed(word_idx)             # (1, embed_size)
        lstm_input = embedding.unsqueeze(1)          # (1, 1, embed_size)
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: (1, 1, hidden_size)
        logits = self.linear(self.dropout(lstm_out.squeeze(1)))    # (1, vocab_size)
        return logits, hidden


# ==========================================
# 3. CACHE AND LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    """Loads the models and vocabulary once to save memory and speed up the app."""
    device = torch.device("cpu")
    
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, dropout=0.3).to(device)
    
    encoder.load_state_dict(torch.load('encoder_weights.pth', map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load('decoder_weights.pth', map_location=device, weights_only=True))
    
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return encoder, decoder, vocab, transform, device


# ==========================================
# 4. INFERENCE FUNCTION
# ==========================================
def clone_hidden(hidden):
    return tuple(state.clone() for state in hidden)


def normalize_caption(tokens):
    filtered_tokens = [
        token for token in tokens
        if token not in {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
    ]

    if not filtered_tokens:
        return "Could not generate a caption."

    caption = " ".join(filtered_tokens)
    caption = caption.replace(" .", ".").replace(" ,", ",")
    caption = caption.replace(" !", "!").replace(" ?", "?")

    if caption and not caption.endswith((".", "!", "?")):
        caption += "."

    return caption[:1].upper() + caption[1:]


def generate_caption(image, encoder, decoder, vocab, transform, device, max_length=20, beam_width=3):
    image_tensor = transform(image).unsqueeze(0).to(device)
    sos_idx = vocab.stoi["<SOS>"]
    eos_idx = vocab.stoi["<EOS>"]
    unk_idx = vocab.stoi.get("<UNK>")

    with torch.no_grad():
        features = encoder(image_tensor)
        initial_hidden = decoder.init_hidden_with_features(features)

        beams = [{
            "tokens": [],
            "last_token": sos_idx,
            "hidden": initial_hidden,
            "score": 0.0,
            "finished": False,
        }]

        for _ in range(max_length):
            candidates = []

            for beam in beams:
                if beam["finished"]:
                    candidates.append(beam)
                    continue

                current_word = torch.tensor([beam["last_token"]], device=device)
                logits, hidden = decoder.decode_step(current_word, beam["hidden"])
                log_probs = torch.log_softmax(logits, dim=1)

                if unk_idx is not None:
                    log_probs[0, unk_idx] = -1e9

                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)

                for log_prob, token_idx in zip(top_log_probs[0], top_indices[0]):
                    token = token_idx.item()
                    next_tokens = beam["tokens"] if token == eos_idx else beam["tokens"] + [token]
                    candidates.append({
                        "tokens": next_tokens,
                        "last_token": token,
                        "hidden": clone_hidden(hidden),
                        "score": beam["score"] + log_prob.item(),
                        "finished": token == eos_idx,
                    })

            beams = sorted(
                candidates,
                key=lambda item: item["score"] / max(1, len(item["tokens"])),
                reverse=True,
            )[:beam_width]

            if all(beam["finished"] for beam in beams):
                break

    best_beam = max(
        beams,
        key=lambda item: item["score"] / max(1, len(item["tokens"])),
    )
    caption_tokens = [vocab.itos[token] for token in best_beam["tokens"]]
    return normalize_caption(caption_tokens)


# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Image Captioner", layout="centered")

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, and the Deep Learning model (ResNet-50 + LSTM) will write a caption for it!")

with st.spinner("Loading AI Models into memory... This may take a few seconds on the first run."):
    encoder, decoder, vocab, transform, device = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Analyzing image features and generating text..."):
            caption = generate_caption(image, encoder, decoder, vocab, transform, device)
            
        st.success("Done!")
        st.markdown(f"### **Prediction:** {caption}")
