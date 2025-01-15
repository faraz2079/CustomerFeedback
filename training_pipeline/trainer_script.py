import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Function for pseudo-label generation
def generate_pseudo_labels(model, tokenizer, texts, device):
    logger.info("Generating pseudo-labels...")
    model.eval()
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pseudo_labels = torch.argmax(outputs.logits, axis=-1).tolist()

    return pseudo_labels


# Function to train the model
def train_model(data_path, is_initial_training):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

    # Load dataset for initial or retraining
    if is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        amazon_dataset = load_dataset("amazon_polarity")
        amazon_texts = amazon_dataset["train"]["content"]
        amazon_labels = amazon_dataset["train"]["label"]

        amazon_train_dataset = TextDataset(amazon_texts, amazon_labels, tokenizer)
    else:
        logger.info(f"Retraining: Loading new unlabeled dataset from {data_path}.")
        with open(data_path, "r") as f:
            data = json.load(f)
        texts = [item["text"] for item in data]

        # Step 1: Generate pseudo-labels for retraining
        logger.info("Generating pseudo-labels with pre-trained model...")
        model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
        model.to(device)

        pseudo_labels = generate_pseudo_labels(model, tokenizer, texts, device)
        retrain_dataset = TextDataset(texts, pseudo_labels, tokenizer)

        # Set up the model for retraining
        model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
        model.to(device)

        retrain_args = TrainingArguments(
            output_dir="./retrain_results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            save_steps=10,
            logging_dir="./retrain_logs",
            logging_steps=10,
        )

        retrainer = Trainer(
            model=model,
            args=retrain_args,
            train_dataset=retrain_dataset
        )
        retrainer.train()

        retrain_model_dir = "/home/bhanu/mobilebert_trained_model"
        model.save_pretrained(retrain_model_dir)
        tokenizer.save_pretrained(retrain_model_dir)
        logger.info(f"Fine-tuned model saved to {retrain_model_dir}")
        return

    # Step 2: Initial training on Amazon Polarity dataset
    logger.info("Training on Amazon Polarity dataset...")
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_steps=10,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=amazon_train_dataset
    )

    trainer.train()

    # Save the initial trained model
    initial_model_dir = "/home/bhanu/mobilebert_trained_model"
    model.save_pretrained(initial_model_dir)
    tokenizer.save_pretrained(initial_model_dir)
    logger.info(f"Initial model saved to {initial_model_dir}")


def main():
    data_path = "./file_processing/inputFile.json"  # Path to your custom unlabeled dataset
    is_initial_training = True  # Set this flag to False for retraining
    train_model(data_path, is_initial_training)


if __name__ == "__main__":
    main()
