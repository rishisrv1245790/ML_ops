from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load NLTK's stopwords
stop_words = set(stopwords.words('english'))

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']

        if not user_input:
            return render_template('index.html', error="Please enter some text.")

        preprocessed_text = preprocess_text(user_input)
        results = sentiment_pipeline(preprocessed_text)

        sentiment = results[0]['label']
        confidence = results[0]['score']

        return render_template('index.html', sentiment=sentiment, confidence=confidence, user_input=user_input)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
