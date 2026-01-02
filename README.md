# Holmes Next-Word Predictor
A Deep Learning–based next-word prediction system built using **FastAPI** for the backend and **Streamlit** for the frontend.  
The application predicts the **next word** in a sentence based on previously typed words, trained on *Sherlock Holmes* literary text.

## About the Project
This project demonstrates how a trained Deep Learning language model (LSTM) can be deployed as a REST API and accessed through a simple web interface.  
It focuses on clean backend–frontend separation and real-world NLP deployment practices.  
The model learns contextual word patterns from classic literature and provides text continuation suggestions similar to a basic writing assistant.

## Features
- Next-word prediction using a Deep Learning language model  
- FastAPI backend for real-time model inference  
- Streamlit-based interactive text continuation frontend  
- Model and vocabulary loading using saved artifacts (`.pt`, `.pkl`)  
- Clean and modular project structure  

## Tech Stack
- **Language:** Python  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Deep Learning:** PyTorch (LSTM)  
- **Data Processing:** Text preprocessing  

## Project Structure
```
Holmes-Next-Word-Predictor/
├── api.py                # FastAPI backend
├── ui.py                 # Streamlit frontend
├── model.ipynb           # Model training notebook
├── nextword_model.pt     # Trained PyTorch model
├── vocab.pkl             # Vocabulary mapping
├── idx2word.pkl          # Reverse vocabulary mapping
├── Sherlock Holmes.txt   # Training dataset
├── .gitignore
└── README.md
```
## How to Run the Project

### 1. Install dependencies
pip install torch fastapi uvicorn streamlit requests

### 2. Start the FastAPI backend
```
uvicorn api:app --reload
```
Backend runs at:
```
http://127.0.0.1:8000
```
Swagger UI:
```
http://127.0.0.1:8000/docs
```
### 3. Run the Streamlit frontend
```
streamlit run ui.py
```
## API Endpoints
POST /predict – Predicts the next word based on input text

## Dataset
The dataset consists of Sherlock Holmes literary text, which provides rich language structure and contextual depth suitable for training a word-level language model.

## Purpose
This project is intended for learning and demonstrating:
- Language modeling fundamentals  
- Deep Learning with PyTorch  
- NLP model deployment using FastAPI  
- Frontend–backend integration with Streamlit  
- End-to-end AI application development  
