import tensorflow as tf
from transformers import MobileBertTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import logging
import boto3

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# AWS S3 Configuration (replace with your own)
S3_BUCKET = "your-s3-bucket-name"
MODEL_OUTPUT_PATH = "models/mobilebert_sentiment_model/"

# Initialize S3 client
s3_client = boto3.client("s3")

# Preprocessing function to tokenize the data
def preprocess_data(texts, labels=None):
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
    return encodings, labels

# Function to train the model
def train_model(data_path, is_initial_training=False):
    logger.info(f"Loading dataset from {data_path}...")

    # If it's initial training, download the Amazon Polarity dataset
    if is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        dataset = load_dataset("amazon_polarity")
        texts = dataset["train"]["content"]
        labels = dataset["train"]["label"]  # Labels available in the initial dataset
    else:
        # Load new dataset from the provided path
        logger.info("Retraining: Loading new unlabeled dataset.")
        with open(data_path, "r") as f:
            data = json.load(f)

        texts = [item["text"] for item in data]
        logger.info(f"Loaded {len(texts)} new texts for retraining.")

        # Load pre-trained model (from initial training)
        model = TFAutoModelForSequenceClassification.from_pretrained("mobilebert_sentiment_model", num_labels=2)
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

        # Generate pseudo-labels using the model
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
        logits = model(inputs).logits
        pseudo_labels = np.argmax(logits.numpy(), axis=-1)  # Assign the predicted label (0 or 1)

        logger.info(f"Generated pseudo-labels for new dataset.")
        labels = pseudo_labels  # Use pseudo-labels for training

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

    # Preprocess the training and validation data
    train_encodings, train_labels = preprocess_data(train_texts, train_labels)
    val_encodings, val_labels = preprocess_data(val_texts, val_labels)

    # Load the MobileBERT model for sequence classification
    model = TFAutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(16)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Train the model
    logger.info("Starting model training with pseudo-labels...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

    # Save the trained model locally
    local_model_dir = "mobilebert_sentiment_model"
    model.save_pretrained(local_model_dir)
    logger.info("Model training complete. Saving model locally.")

    # Upload the trained model to S3
    logger.info("Uploading trained model to S3...")
    for file_name in os.listdir(local_model_dir):
        s3_client.upload_file(
            os.path.join(local_model_dir, file_name),
            S3_BUCKET,
            f"{MODEL_OUTPUT_PATH}{file_name}"
        )
    logger.info(f"Model successfully uploaded to s3://{S3_BUCKET}/{MODEL_OUTPUT_PATH}")

# Main function
def main():
    data_path = "new_data.json"  # Path to the new dataset or S3 location
    is_initial_training = False  # Set to True for initial training

    # Train the model (initial or retraining)
    train_model(data_path, is_initial_training)

if __name__ == "__main__":
    main()
