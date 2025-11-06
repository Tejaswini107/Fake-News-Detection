# Fake News Detection System

## Overview
This project detects fake news headlines using natural language processing and machine learning. It converts textual data into meaningful embeddings and trains a Convolutional Neural Network (CNN) to classify news as genuine or fake. A clean, interactive UI allows real-time predictions and logging for future improvements.

## Key Features
- Converts textual data to **Word2Vec embeddings**.
- Trains a **CNN model** to detect semantic patterns in headlines.
- Provides a **responsive UI** with **Gradio** for user-friendly interaction.
- Allows users to **flag predictions**, storing the news article, prediction, and timestamp.
- Tested with **COVID-related misinformation datasets** and verified articles from sources like **WHO**.

## Tools & Technologies
- Python
- TensorFlow & Keras
- Word2Vec
- NLP (Natural Language Processing)
- CNN
- Scikit-learn
- Gradio

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
2. Install required dependencies:
   ```bash
   pip install tensorflow keras gensim scikit-learn gradio
3. Run the application:
  ```bash
   python app_gradio.py
