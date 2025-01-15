import tensorflow as tf
from transformers import MobileBertTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import argparse
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_data(texts, labels=None):
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
    return encodings, labels

def train_model(args):
    # Load dataset
    if args.is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        dataset = load_dataset("amazon_polarity")
        texts = dataset["train"]["content"]
        labels = dataset["train"]["label"]
    else:
        logger.info(f"Retraining: Loading new dataset from {args.data_dir}/inputFile.json.")
        with open(f"{args.data_dir}/inputFile.json", "r") as f:
            data = json.load(f)

        texts = [item["text"] for item in data]

        # Load pre-trained model
        model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

        # Generate pseudo-labels
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
        logits = model(inputs).logits
        pseudo_labels = np.argmax(logits.numpy(), axis=-1)

        logger.info("Pseudo-labels generated for new dataset.")
        labels = data.get("label", pseudo_labels)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # Preprocess data
    train_encodings, train_labels = preprocess_data(train_texts, train_labels)
    val_encodings, val_labels = preprocess_data(val_texts, val_labels)

    # Load model
    model = TFAutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(args.train_batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(args.eval_batch_size)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs)

    # Save the trained model
    output_dir = "/opt/ml/model"
    model.save_pretrained(output_dir)
    logger.info(f"Model training complete. Model saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model_name", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--is_initial_training", type=int, default=1)  # 1 for True, 0 for False

    args = parser.parse_args()
    train_model(args)
