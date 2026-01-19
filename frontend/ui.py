import streamlit as st
import requests

# CONFIG
API_URL = "https://literature-next-word-predictor.onrender.com/predict"  

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# UI
st.title("🧠 Next Word Prediction")
st.markdown(
    """
    **LSTM-based Language Model**  
    Predicts the **next word** using context, POS embeddings & attention.
    """
)

text = st.text_area(
    "Enter text (minimum 25 words):",
    height=150,
    placeholder="The night was cold and the wind blew softly through the trees as the traveler continued his long journey..."
)

temperature = st.slider(
    "Temperature (controls randomness)",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1
)

predict_btn = st.button("🔮 Predict Next Word")


# REQUEST
if predict_btn:
    if len(text.split()) < 25:
        st.error("❌ Please enter at least **25 words**.")
    else:
        payload = {
            "text": text,
            "temperature": float(temperature)
        }

        try:
            with st.spinner("Predicting..."):
                response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code != 200:
                st.error(f"❌ API Error: {response.text}")
            else:
                result = response.json()

                st.subheader("📊 Top-5 Predictions")

                for i, item in enumerate(result["top_5_predictions"], start=1):
                    st.write(
                        f"**{i}. {item['word']}** — probability `{item['prob']:.3f}`"
                    )

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI server. Is it running?")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# FOOTER
st.markdown("---")
st.caption(
    "Built with PyTorch · BiLSTM + Attention · POS-aware · Top-5 decoding"
)
