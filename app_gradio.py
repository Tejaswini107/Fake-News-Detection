import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("fake_news_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set maximum length (use the same as used during training)
maxlen = 500

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_news(news_text):
    cleaned = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    if pred >= 0.5:
        return f"🔵 Real News ({pred:.2f})"
    else:
        return f"🔴 Fake News ({pred:.2f})"

# Gradio Interface
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, label="Enter News Article"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector",
    description="Enter a news article or paragraph to check if it's fake or real."
)

iface.launch()
