import torch
from torch.utils.data import Dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments, MobileBertConfig
from datasets import load_dataset
import json
import logging
import boto3
import os
from botocore.exceptions import ClientError
from textblob import TextBlob
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# AWS S3 Configuration
S3_BUCKET = "customerfeedbackmlbucket"
MODEL_PATH = "models/"
NEW_DATA_PATH = "datasets/"

s3_client = boto3.client('s3', region_name='eu-central-1')

trainer_dir = os.path.expanduser("~/trainerModel/mobilebert_trained_model")
result_dir = os.path.expanduser("~/trainerModel/results")
dataset_dir = os.path.expanduser("~/trainerModel/input")
logs_dir = os.path.expanduser("~/trainerModel/logs")
os.makedirs(trainer_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

def encode_labels(labels, num_classes):
    multi_hot_labels = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label_list in enumerate(labels):
        for label in label_list:
            multi_hot_labels[i, label] = 1.0
    return multi_hot_labels

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, num_classes, max_length=256):
        self.texts = texts
        self.labels = encode_labels(labels, num_classes)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels is not None else -1
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                                   return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def relabel_data(example):
    logger.info(f"Context: {example['content']}")
    logger.info(f"Initial label is: {example['label']}")
    content = example["content"].lower()
    if not content.strip():
        logger.info("Empty content encountered.")
        example["label"] = 2
        logger.info(f"Label Assigned: {example['label']}")
        return example
    sentences = TextBlob(content).sentences
    sentiment_score = sum(s.sentiment.polarity for s in sentences) / len(sentences)
    if sentiment_score < -0.6:  # Strongly negative
        example["label"] = 0
    elif -0.6 <= sentiment_score < -0.1:  # Negative
        example["label"] = 1
    elif -0.1 <= sentiment_score <= 0.2:  # Neutral
        example["label"] = 2
    elif 0.2 < sentiment_score <= 0.6:  # Positive
        example["label"] = 3
    elif 0.6 < sentiment_score <= 1.0:  # Strongly positive
        example["label"] = 4
    else:
        raise ValueError(f"Sentiment score out of range: {sentiment_score}")
    logger.info(f"Label Assigned: {example['label']}")
    return example

# Function for pseudo-label generation
def generate_pseudo_labels(model, tokenizer, texts, device, confidence_threshold=0.9):
    logger.info("Generating pseudo-labels...")
    model.eval()
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_probs, pseudo_labels = torch.max(probabilities, dim=-1)

    # Filter confident labels
    confident_indices = (max_probs >= confidence_threshold).cpu().numpy().tolist()
    texts_confident = [text for i, text in enumerate(texts) if confident_indices[i] == 1]
    pseudo_labels_confident = [pseudo_labels[i].item() for i, is_confident in enumerate(confident_indices) if
                               is_confident]

    # Fallback for non-confident labels
    texts_non_confident = [text for i, text in enumerate(texts) if confident_indices[i] == 0]
    pseudo_labels_non_confident = []
    for text in texts_non_confident:
        example = {"content": text}
        pseudo_label = relabel_data(example)["label"]
        pseudo_labels_non_confident.append(pseudo_label)

    # Combine results
    texts_final = texts_confident + texts_non_confident
    pseudo_labels_final = pseudo_labels_confident + pseudo_labels_non_confident
    return texts_final, pseudo_labels_final

# Function to train the model
def train_model(data_path, is_initial_training):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 5
    id2label = {0: "VERY_NEGATIVE", 1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE", 4: "VERY_POSITIVE"}
    label2id = {label: idx for idx, label in id2label.items()}
    config = MobileBertConfig.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )
    # Load default dataset for initial training else retrain on new dataset
    if is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        amazon_dataset = load_dataset("amazon_polarity")
        logger.info(f"Printing Amazon Dataset: {amazon_dataset}")
        amazon_dataset = amazon_dataset.map(relabel_data)
        amazon_texts = amazon_dataset["train"]["content"]
        amazon_labels = amazon_dataset["train"]["label"]
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased", model_max_length=256)
        amazon_train_dataset = TextDataset(amazon_texts, amazon_labels, tokenizer, num_classes=num_labels)
        logger.info("Training on Amazon Polarity dataset...")
        model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=config)
        model.to(device)
        training_args = TrainingArguments(
            output_dir=f"{result_dir}",
            save_steps=1000,
            save_total_limit=2,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            logging_dir=f"{logs_dir}",
            logging_steps=2000,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=amazon_train_dataset
        )

        trainer.train()

        # Save the initial trained model
        initial_model_dir = f"{trainer_dir}"
        model.save_pretrained(initial_model_dir)
        tokenizer.save_pretrained(initial_model_dir)
        for file_name in os.listdir(initial_model_dir):
            local_file_path = os.path.join(initial_model_dir, file_name)
            s3_key = f"{MODEL_PATH}{file_name}"
            s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
            logger.info(f"Uploaded {local_file_path} to s3://{S3_BUCKET}/{s3_key}")
        logger.info(f"Initial model saved to {initial_model_dir}")

    else:
        logger.info(f"Retraining: Loading new unlabeled dataset from {data_path}.")
        tokenizer = MobileBertTokenizer.from_pretrained(f"{trainer_dir}")
        texts = []
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])
        logger.info("Generating pseudo-labels with pre-trained model...")
        model = MobileBertForSequenceClassification.from_pretrained(f"{trainer_dir}")
        model.to(device)
        confident_texts, pseudo_labels = generate_pseudo_labels(model, tokenizer, texts, device,0.9)
        if len(confident_texts) == 0:
            logger.warning("No confident pseudo-labels generated. Aborting retraining.")
            return
        logger.info(f"Retained {len(confident_texts)} out of {len(texts)} samples for retraining.")
        retrain_dataset = TextDataset(confident_texts, pseudo_labels, tokenizer)

        retrain_args = TrainingArguments(
            output_dir=f"{result_dir}",
            save_steps=1000,
            save_total_limit=2,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            logging_dir=f"{logs_dir}",
            logging_steps=2000,
        )

        retrainer = Trainer(
            model=model,
            args=retrain_args,
            train_dataset=retrain_dataset
        )
        retrainer.train()

        retrain_model_dir = f"{trainer_dir}"
        model.save_pretrained(retrain_model_dir)
        tokenizer.save_pretrained(retrain_model_dir)
        for file_name in os.listdir(retrain_model_dir):
            local_file_path = os.path.join(retrain_model_dir, file_name)
            s3_key = f"{MODEL_PATH}{file_name}"
            s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
            logger.info(f"Uploaded {local_file_path} to s3://{S3_BUCKET}/{s3_key}")

        logger.info(f"Fine-tuned model saved to {retrain_model_dir}")
        return

def main():
    data_path = f"{dataset_dir}"
    dataset = os.path.join(data_path, "inputFile.jsonl")
    try:
        s3_client.download_file(S3_BUCKET, f"{NEW_DATA_PATH}inputFile.jsonl", dataset)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.error("File not found in S3 (404). Skipping download.")
        else:
            logger.error(f"Unexpected S3 error: {e}")
            raise
    except Exception as ex:
        logger.error(f"Unexpected error occured: {ex}")
        raise
    is_initial_training = True  # Set this flag to False for retraining
    train_model(dataset, is_initial_training)

if __name__ == "__main__":
    main()
