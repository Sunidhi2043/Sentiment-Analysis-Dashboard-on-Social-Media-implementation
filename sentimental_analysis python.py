from flask import Flask, render_template, request, jsonify
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Route for the main page
@app.route("/")
def home():
    return render_template("frontpage.html")

# Route for sentiment analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    social_media_text = data.get("text")

    if not social_media_text:
        return jsonify({"error": "No text provided"}), 400

    # Perform sentiment analysis
    sentiment, score = predict_sentiment(social_media_text)
    return jsonify({"sentiment": sentiment, "score": f"{score:.2f}%"})

# Load and preprocess data for training
nltk.download('punkt')
nltk.download('stopwords')

# CSV file path
file_path = r"C:\\Users\\nimit\\Downloads\\twitter_training.csv"

# Reading CSV file
data = pd.read_csv(file_path, encoding='latin1')

# Handle missing data: Remove rows with missing data
data = data.dropna(subset=['text', 'label'])

# Remove any classes with less than 2 samples
class_counts = data['label'].value_counts()
valid_classes = class_counts[class_counts > 1].index
data = data[data['label'].isin(valid_classes)]

# Separate text and labels 
texts, labels = data['text'], data['label']

# Convert text labels to 0 and 1
label_map = {'negative': 0, 'positive': 1}
labels = labels.map(label_map)

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, float) or isinstance(text, type(None)):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess all texts
preprocessed_texts = [preprocess_text(text) for text in texts]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict sentiment for inputs
def predict_sentiment(text):
    preprocessed = preprocess_text(text)
    vectorized = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)

    predicted_class = prediction[0]
    score = probabilities[0][predicted_class]

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment, score * 100

if __name__ == "__main__":
    app.run(debug=True)
