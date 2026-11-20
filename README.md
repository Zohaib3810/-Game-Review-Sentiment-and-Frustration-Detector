# -Game-Review-Sentiment-and-Frustration-Detector
An NLP-based project developed to analyze mobile game reviews, determine sentiment, and identify player frustration points (such as monetization/IAP issues and performance bugs). This repository includes the complete data preprocessing pipeline, machine learning and deep learning models, performance comparisons, and an interactive Streamlit Web UI.
---

## Project Overview

Mobile game reviews contain crucial feedback for developers, but manual analysis at scale is impossible. This project automates sentiment classification on a real-world dataset of Google Play Store reviews for Genshin Impact, classifiying them into Positive and Negative categories. It compares classical machine learning models with a deep learning recurrent neural network (RNN).

## Dataset

- **Source:** Real-world dataset of Google Play Store reviews for Genshin Impact (`genshin_review.csv`).
- **Size:** 995 reviews.
- **Classes:** Positive (658 reviews) and Negative (337 reviews).
- **Mapping:** User ratings (1 to 5 stars) are converted to sentiment classes. Ratings of 3 stars and below are mapped to "Negative", and ratings of 4 to 5 stars are mapped to "Positive".

---

## NLP Preprocessing Pipeline

To clean and normalize the unstructured review text, the following pipeline is implemented using NLTK:

1. **Case Normalization:** Convert all text to lowercase.
2. **URL and Noise Removal:** Erase web links and special characters.
3. **Tokenization:** Split sentences and word tokens.
4. **Stop-word Filtering:** Remove common English stopwords while preserving sentiment-carrying words like 'no', 'not', 'never', 'cannot', 'cant', 'dont'.
5. **Lemmatization:** Reduce words to their base forms using WordNetLemmatizer (e.g., "played" and "plays" become "play").

---

## Feature Extraction & Modeling

### 1. TF-IDF + Classical Machine Learning
Text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams ranging from unigrams to trigrams (1, 3) and limited to the top 1000 features.
- **Model A: Naive Bayes (MultinomialNB)**
- **Model B: Support Vector Machine (LinearSVC)**

### 2. Recurrent Neural Network (RNN)
- **Architecture:** Bidirectional LSTM network built using TensorFlow/Keras.
- **Features:** Word embeddings, Spatial Dropout, Bidirectional LSTM layer, Global Max Pooling, Dense layers with L2 regularization, and Dropout.
- **Optimization:** Trained using Adam optimizer, class weight balancing to handle data imbalance, and early stopping callbacks based on validation loss.

---

## Results and Performance Comparison

Below is the performance comparison of the models evaluated on the test split (20%):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Naive Bayes | 0.7538 | 0.7465 | 0.7538 | 0.7381 |
| SVM | 0.7337 | 0.7253 | 0.7337 | 0.7271 |
| Improved Bidirectional RNN | 0.7286 | 0.7307 | 0.7286 | 0.7296 |

---

## Streamlit Web UI

An interactive web interface is developed using Streamlit, allowing users to select a classification model, input their own game reviews, and get real-time sentiment predictions.

### How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/game-review-sentiment.git
   cd game-review-sentiment
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

### Files in the Repository
- `app.py`: The Streamlit web application code.
- `genshin_review.csv`: The real-world reviews dataset.
- `tfidf.pkl` / `nb_model.pkl` / `svm_model.pkl`: Saved classical NLP pipeline elements and models.
- `rnn_model.keras` / `tokenizer.pkl`: Saved deep learning model and associated text tokenizer.
```
