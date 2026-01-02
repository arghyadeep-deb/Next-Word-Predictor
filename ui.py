import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Holmes Next-Word Predictor")

# -------------------------
# TITLE & DESCRIPTION
# -------------------------
st.title("🕵️ Holmes Next-Word Predictor")
st.write(
    "This model is trained on *Sherlock Holmes* text and predicts the **next word** "
    "based on the last few words you type."
)

st.divider()

# -------------------------
# TEXT INPUT
# -------------------------
text = st.text_area(
    "Write a sentence",
    placeholder="Example: to sherlock holmes she",
    height=120
)

st.caption("Minimum 4 words required")

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Next Word"):
    if text.strip() == "":
        st.warning("Please write something first.")
    else:
        response = requests.post(API_URL, json={"text": text})

        if response.status_code == 200:
            result = response.json()

            if "predicted_next_word" in result:
                predicted_word = result["predicted_next_word"]

                st.subheader("🧠 Suggested Continuation")
                st.write(text + " **" + predicted_word + "**")
            else:
                st.error(result["error"])
        else:
            st.error("FastAPI server is not running.")