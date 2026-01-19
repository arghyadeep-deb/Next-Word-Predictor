# Next Word Predictor  
An LSTM-based Language Model that predicts the **next probable word** given an input text sequence using Natural Language Processing (NLP) and Deep Learning techniques.  
The system is deployed end-to-end using **FastAPI** for the backend and **Streamlit** for the frontend.
---
## Live Deployment

- 🔗 **Frontend (Streamlit App):**
```
https://liteerature-next-word-predictor.streamlit.app/
```
- 🔗 **Backend API (FastAPI):**
```
https://literature-next-word-predictor.onrender.com/docs
```
The Streamlit frontend communicates with a FastAPI backend deployed on Render,  
forming a production-style deep learning inference pipeline for next-word prediction.
---
## About the Project
This project demonstrates how a **deep learning–based language model** can be trained, evaluated, and deployed as a real-time application.  
It focuses on **next-word prediction** using a **BiLSTM with Attention and POS embeddings**, enabling the model to learn contextual and grammatical patterns from large text corpora.
The project showcases a complete **ML → API → UI deployment pipeline**, making it suitable for production-grade demonstrations and portfolio use.
---

## Features
- Next-word prediction using LSTM-based language modeling  
- Word-level tokenization with controlled vocabulary size  
- POS-tag embeddings to improve grammatical awareness  
- BiLSTM with Attention mechanism for contextual learning  
- Top-1 and Top-5 accuracy evaluation  
- Perplexity tracking for language model quality  
- Temperature-based sampling for better text generation  
- FastAPI backend for real-time inference  
- Streamlit interactive frontend  
- Pretrained model and vocab loading from saved artifacts  
- Clean, modular, and scalable project structure  
---

## Tech Stack
- **Language:** Python  
- **Deep Learning:** PyTorch  
- **NLP:** NLTK, POS Tagging  
- **Model Architecture:** BiLSTM + Attention  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Utilities:** NumPy, Pickle  
---

## Project Structure
```
Next-Word-Predictor/
│
├── backend/
│   └── api.py                  (FastAPI backend)
│
├── frontend/
│   └── ui.py                   (Streamlit frontend)
│
├── model/
│   ├── best_model.pt           (Best trained model)
│   ├── final_model.pt          (Final model checkpoint)
│   ├── vocab.pkl               (Word vocabulary)
│   ├── pos_vocab.pkl           (POS tag vocabulary)
│   ├── train_losses.pkl
│   ├── val_losses.pkl
│   ├── train_top1.pkl
│   ├── val_top1.pkl
│   ├── train_top5.pkl
│   └── val_top5.pkl
│
├── requirements.txt            (Project dependencies)
├── .gitignore
└── README.md
```
---

## How to Run the Project
### Step 1: Install dependencies
```
pip install -r requirements.txt
```
---

### Step 2: Start the FastAPI backend
```
cd backend
uvicorn api:app --reload
```

#### Backend URL
```
http://127.0.0.1:8000
```

#### Health Check
```
GET /health
```

#### Swagger UI
```
http://127.0.0.1:8000/docs
```
---

### Step 3: Run the Streamlit frontend
```
cd frontend
streamlit run ui.py
```
---

## API Endpoints
### `GET /health`
Checks whether the API and model are loaded successfully.

### `POST /predict`
Predicts the **next word** based on the input text sequence.

**Request Body**
```
{
  "text": "the night was cold and the wind moved slowly through the empty street",
  "temperature": 1.0
}
```

**Response**
```json
{
  "next_word": "while",
  "top_predictions": ["while", "as", "when", "and", "because"]
}
```
---

## Model Evaluation
- **Top-1 Accuracy:** ~21%  
- **Top-5 Accuracy:** ~42%  
- **Validation Perplexity:** ~120  
These metrics are strong for a **word-level LSTM language model** trained on a limited corpus and demonstrate meaningful contextual learning.
---

## Dataset
The model is trained on a curated corpus of classic English literature, enabling it to learn:
- Long-range dependencies  
- Grammar-aware word prediction  
- Contextual continuity in sentences  
The dataset size is carefully balanced to achieve good performance without excessive training time.
---

## Purpose
This project is intended to demonstrate:
- Deep Learning–based language modeling  
- POS-aware NLP feature engineering  
- Training and evaluating LSTM models  
- Perplexity-based model assessment  
- ML model deployment using FastAPI  
- Frontend–backend integration with Streamlit  
- Production-style ML project structuring  
---
