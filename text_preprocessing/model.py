import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import load_data, prepare_tokenizer, preprocess_texts

def create_model(num_classes):
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=128, input_length=500),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Number of classes based on your dataset
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(file_path):
    # Load data
    (x_train, y_train), (x_test, y_test), label_mapping = load_data(file_path)  # Replace with your actual CSV file path

    # Prepare tokenizer and preprocess texts
    tokenizer = prepare_tokenizer(x_train)
    x_train_padded = preprocess_texts(tokenizer, x_train)
    x_test_padded = preprocess_texts(tokenizer, x_test)

    # Start MLflow experiment
    mlflow.start_run()

    # Create and train model
    model = create_model(len(label_mapping))  # Pass number of classes
    model.fit(x_train_padded, y_train, epochs=10, validation_split=0.1)

    # Log model and metrics
    mlflow.tensorflow.log_model(model, "model")
    mlflow.log_metric("accuracy", model.evaluate(x_test_padded, y_test)[1])

    # Save the tokenizer and label mapping for later use
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('label_mapping.pickle', 'wb') as handle:
        pickle.dump(label_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mlflow.end_run()

if __name__ == "__main__":
    train_model('your_dataset.csv')  # Replace with your actual CSV file path
