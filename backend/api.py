from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re
import os

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# -------------------------
# LOAD VOCABS
# -------------------------
with open(os.path.join(MODEL_DIR, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(MODEL_DIR, "pos_vocab.pkl"), "rb") as f:
    pos_vocab = pickle.load(f)

idx2word = {i: w for w, i in vocab.items()}

PAD_IDX = vocab.get("<PAD>", 0)
UNK_IDX = vocab.get("<UNK>", 1)

VOCAB_SIZE = len(vocab)
POS_SIZE = len(pos_vocab)

SEQ_LEN = 25

# -------------------------
# MODEL (EXACT MATCH)
# -------------------------
class BiLSTMAttn(nn.Module):
    def __init__(self):
        super().__init__()

        self.word_emb = nn.Embedding(VOCAB_SIZE, 200, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(POS_SIZE, 32)

        self.lstm = nn.LSTM(
            input_size=232,      # 200 + 32
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.ln = nn.LayerNorm(512)

        self.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, VOCAB_SIZE)

    def forward(self, xw, xp):
        w = self.word_emb(xw)
        p = self.pos_emb(xp)
        x = torch.cat([w, p], dim=-1)

        h, _ = self.lstm(x)
        h = self.ln(h)

        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        ctx = (a.unsqueeze(-1) * h).sum(1)

        return self.fc(ctx)

# -------------------------
# LOAD MODEL
# -------------------------
model = BiLSTMAttn().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device)
)
model.eval()

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI(title="Next Word Predictor API")

class InputText(BaseModel):
    text: str
    temperature: float = 1.0

# -------------------------
# UTILS
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def dummy_pos(seq_len):
    # During inference, POS is unknown → use NOUN (or 0)
    return torch.zeros(seq_len, dtype=torch.long)

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "vocab_size": VOCAB_SIZE,
        "pos_vocab_size": POS_SIZE
    }

# -------------------------
# PREDICTION
# -------------------------
@app.post("/predict")
def predict(data: InputText):
    text = clean_text(data.text)
    words = text.split()

    if len(words) < SEQ_LEN:
        return {"error": f"Minimum {SEQ_LEN} words required"}

    words = words[-SEQ_LEN:]

    xw = torch.tensor(
        [[vocab.get(w, UNK_IDX) for w in words]],
        dtype=torch.long
    ).to(device)

    xp = dummy_pos(SEQ_LEN).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(xw, xp)
        logits /= max(data.temperature, 1e-3)
        probs = torch.softmax(logits, dim=1)

        top5_prob, top5_idx = probs.topk(5, dim=1)

    preds = [
        {"word": idx2word[i.item()], "prob": round(p.item(), 4)}
        for i, p in zip(top5_idx[0], top5_prob[0])
    ]

    return {"top_5_predictions": preds}
