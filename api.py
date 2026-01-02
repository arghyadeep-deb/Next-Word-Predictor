from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD VOCAB FILES
# -------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(vocab)

# -------------------------
# MODEL DEFINITION
# -------------------------
class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# -------------------------
# LOAD MODEL
# -------------------------
EMBED_DIM = 100
HIDDEN_DIM = 128
SEQ_LEN = 4

model = NextWordModel(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("nextword_model.pt", map_location=device))
model.eval()

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI()

class InputText(BaseModel):
    text: str

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -------------------------
# PREDICT API
# -------------------------
@app.post("/predict")
def predict(data: InputText):
    text = clean_text(data.text)
    words = text.split()

    if len(words) < SEQ_LEN:
        return {"error": "Please enter at least 4 words"}

    words = words[-SEQ_LEN:]

    encoded = torch.tensor(
        [[vocab.get(w, vocab["<UNK>"]) for w in words]],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        output = model(encoded)
        pred_idx = output.argmax(dim=1).item()

    return {
        "input_text": data.text,
        "predicted_next_word": idx2word[pred_idx]
    }
