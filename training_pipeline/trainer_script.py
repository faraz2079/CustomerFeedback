import torch
from torch.utils.data import Dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments, MobileBertConfig
from datasets import load_dataset, ClassLabel
import json
import logging
import boto3
import os
from botocore.exceptions import ClientError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

analyzer = SentimentIntensityAnalyzer()

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, num_classes: int, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.max_length = max_length
        for label in self.labels:
            if label < 0 or label >= self.num_classes:
                raise ValueError(f"Label {label} out of range. It must be between 0 and {self.num_classes - 1}.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                                   return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

def relabel_data(example):
    logger.info(f"Content is: {example['content']}")
    content = example["content"].strip().lower()
    if not content:
        logger.info("Empty content encountered.")
        example["label"] = 2
        logger.info(f"Label Assigned for Empty content: {example['label']}")
        return example
    sentiment_score = get_vader_sentiment(content)
    logger.info(f"After calculating the sentiment score: {sentiment_score}")
    if sentiment_score < -0.5:  # VERY NEGATIVE
        example["label"] = 0
    elif -0.5 <= sentiment_score < -0.1:  # NEGATIVE
        example["label"] = 1
    elif -0.1 <= sentiment_score <= 0.1:  # NEUTRAL
        example["label"] = 2
    elif 0.1 < sentiment_score <= 0.5:  # POSITIVE
        example["label"] = 3
    elif 0.5 < sentiment_score <= 1.0:  # VERY POSITIVE
        example["label"] = 4
    logger.info(f"Label Assigned for the content based on sentiment score: {example['label']}")
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
    confident_indices = (max_probs >= confidence_threshold).nonzero(as_tuple=True)[0].cpu().numpy()
    texts_confident = [texts[i] for i in confident_indices]
    pseudo_labels_confident = [pseudo_labels[i].item() for i in confident_indices]

    # Fallback for non-confident labels
    texts_non_confident = [text for i, text in enumerate(texts) if i not in confident_indices]
    pseudo_labels_non_confident = [relabel_data({"content": text})["label"] for text in texts_non_confident]
    # Combine results
    texts_final = texts_confident + texts_non_confident
    pseudo_labels_final = pseudo_labels_confident + pseudo_labels_non_confident
    logger.info(f"Final Text is: {texts_final} and Final pseudo label generated is: {pseudo_labels_final}")
    return texts_final, pseudo_labels_final

# Function to train the model
def train_model(data_path, is_initial_training):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = {0: "VERY NEGATIVE", 1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE", 4: "VERY POSITIVE"}
    label2id = {label: idx for idx, label in id2label.items()}
    config = MobileBertConfig.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=5,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification"
    )
    # Load default dataset for initial training else retrain on new dataset
    if is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        new_labels = ClassLabel(num_classes=5, names=["VERY NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY POSITIVE"])
        amazon_dataset = load_dataset("amazon_polarity")
        amazon_dataset = amazon_dataset.cast_column("label", new_labels)
        amazon_dataset = amazon_dataset.map(relabel_data)
        amazon_texts = amazon_dataset["train"]["content"]
        amazon_labels = amazon_dataset["train"]["label"]
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased", model_max_length=256)
        amazon_train_dataset = TextDataset(amazon_texts, amazon_labels, tokenizer, num_classes=5, max_length=256)
        logger.info("Training on Amazon Polarity dataset...")
        model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=config)
        model.to(device)
        training_args = TrainingArguments(
            output_dir=f"{result_dir}",
            save_steps=1000,
            save_total_limit=2,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=100,
            num_train_epochs=3,
            logging_dir=f"{logs_dir}",
            logging_steps=1000,
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
        retrain_dataset = TextDataset(confident_texts, pseudo_labels, tokenizer, num_classes=5, max_length=256)

        retrain_args = TrainingArguments(
            output_dir=f"{result_dir}",
            save_steps=1000,
            save_total_limit=2,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=100,
            num_train_epochs=3,
            logging_dir=f"{logs_dir}",
            logging_steps=1000,
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
