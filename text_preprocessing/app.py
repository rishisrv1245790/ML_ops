from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import mlflow.tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model, tokenizer, and label mapping
model = mlflow.tensorflow.load_model("model")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_mapping.pickle', 'rb') as handle:
    label_mapping = pickle.load(handle)

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=500)
    return padded

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        prediction = model.predict(processed_text)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        
        # Reverse the label mapping to get the class name
        predicted_class = {v: k for k, v in label_mapping.items()}[predicted_class_index]
        
        return render_template('result.html', predicted_class=predicted_class)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
