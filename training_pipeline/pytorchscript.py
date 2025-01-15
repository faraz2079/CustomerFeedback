import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import logging
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Preprocessing function
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
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train_model(data_path, is_initial_training=False):
    logger.info(f"Loading dataset from {data_path}...")

    # Load initial or retraining dataset
    if is_initial_training:
        logger.info("Initial training: Loading Amazon Polarity dataset.")
        dataset = load_dataset("amazon_polarity")
        texts = dataset["train"]["content"]
        labels = dataset["train"]["label"]
    else:
        logger.info("Retraining: Loading new unlabeled dataset.")
        with open(data_path, "r") as f:
            data = json.load(f)
        texts = [item["text"] for item in data]

        # Generate pseudo-labels
        logger.info("Generating pseudo-labels with pre-trained model.")
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        model = MobileBertForSequenceClassification.from_pretrained("/home/bhanu/mobilebert_sentiment_model")
        model.eval()

        inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pseudo_labels = torch.argmax(outputs.logits, axis=-1).tolist()

        labels = data.get("label", pseudo_labels)

    # Split into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    # Define DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize the model
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model.train()

    # Define optimizer, loss function, and learning rate
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    # Training loop
    logger.info("Starting model training...")
    writer = SummaryWriter("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):  # Number of epochs
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        logger.info(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, axis=-1)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_dataset)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained model
    local_model_dir = "/home/bhanu/mobilebert_sentiment_model"
    model.save_pretrained(local_model_dir)
    logger.info("Model training complete. Saving model locally.")

# Main function
def main():
    data_path = "inputFile.json"
    is_initial_training = True
    train_model(data_path, is_initial_training)

if __name__ == "__main__":
    main()
