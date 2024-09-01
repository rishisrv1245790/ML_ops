import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    # Load your dataset
    df = pd.read_csv(file_path)

    # Ensure columns are named correctly
    df.columns = ['text', 'label']

    # Map labels to integers
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_mapping)

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test), label_mapping

def prepare_tokenizer(x_train):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(x_train)
    return tokenizer

def preprocess_texts(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=500)
    return padded
