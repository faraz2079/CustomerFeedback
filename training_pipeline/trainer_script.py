import torch
from torch.utils.data import Dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json
import logging
import boto3
import os
from botocore.exceptions import ClientError

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

# Custom Dataset Class
class TextDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length=256):
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
def generate_pseudo_labels(model, tokenizer, texts, device, confidence_threshold=0.9):
	logger.info("Generating pseudo-labels...")
	model.eval()
	inputs = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
	inputs = {key: val.to(device) for key, val in inputs.items()}

	with torch.no_grad():
		outputs = model(**inputs)
		probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
		max_probs, pseudo_labels = torch.max(probabilities, dim=-1)

	# Filter labels by confidence
	confident_indices = (max_probs >= confidence_threshold).cpu().numpy()
	confident_indices = confident_indices.tolist()
	pseudo_labels = pseudo_labels.tolist()
	texts = [text for i, text in enumerate(texts) if confident_indices[i] == 1]
	pseudo_labels = [pseudo_labels[i] for i in range(len(confident_indices)) if confident_indices[i] == 1]

	return texts, pseudo_labels


# Function to train the model
def train_model(data_path, is_initial_training):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load default dataset for initial training else retrain on new dataset
	if is_initial_training:
		logger.info("Initial training: Loading Amazon Polarity dataset.")
		amazon_dataset = load_dataset("amazon_polarity")
		amazon_texts = amazon_dataset["train"]["content"]
		amazon_labels = amazon_dataset["train"]["label"]
		tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
		amazon_train_dataset = TextDataset(amazon_texts, amazon_labels, tokenizer)
		logger.info("Training on Amazon Polarity dataset...")
		model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=5)
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
		model = MobileBertForSequenceClassification.from_pretrained(f"{trainer_dir}", num_labels=5)
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
		logger.error(f"Error downloading inputFile.jsonl from S3: {e}")
		if e.response["Error"]["Code"] == "404":
			logger.error("File not found in S3 (404). Skipping download.")
		else:
			logger.error(f"Unexpected S3 error: {e}")
			raise
	is_initial_training = True  # Set this flag to False for retraining
	train_model(dataset, is_initial_training)

if __name__ == "__main__":
	main()
