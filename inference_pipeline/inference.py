from fastapi import FastAPI, HTTPException
from feedback_request_model import FeedbackRequest
from feedback_response_model import FeedbackResponse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
import os
import psutil
import pyRAPL
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# AWS S3 Configuration
S3_BUCKET = "customerfeedbackmlbucket"
MODEL_PATH = "models/"
NEW_DATA_PATH = "datasets/"

s3_client = boto3.client('s3', region_name='eu-central-1')

# Load the model and tokenizer from S3
def download_model_from_s3():
    local_model_dir = os.path.expanduser("~/s3/inference/models/")
    os.makedirs(local_model_dir, exist_ok=True)

    logger.info("Downloading model files from S3...")
    try:
        # List all files in the specified S3 bucket directory
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PATH)
        if 'Contents' not in response:
            raise ValueError(f"No files found in S3 path: {MODEL_PATH}")

        for obj in response['Contents']:
            file_name = os.path.basename(obj['Key'])
            if file_name:
                local_file_path = os.path.join(local_model_dir, file_name)
                try:
                    s3_client.download_file(S3_BUCKET, obj['Key'], local_file_path)
                    logger.info(f"Successfully downloaded {file_name} from S3.")
                except Exception as e:
                    logger.error(f"Error downloading {file_name} from S3: {e}")
                    raise e
    except Exception as e:
        logger.error(f"Error listing files from S3: {e}")
        raise e
    logger.info("Model download complete.")
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    return model, tokenizer

try:
    model, tokenizer = download_model_from_s3()
except Exception as e:
    logger.critical("Failed to load model. Service cannot start.", exc_info=True)
    raise RuntimeError("Model initialization failed.") from e

# Define sentiment labels
sentiment_labels = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

# Log new data to S3
def log_new_data_to_s3(feedback):
    new_data = {
        "text": feedback.text,
        "stars": feedback.stars
    }
    new_data_path = os.path.expanduser("~/s3/inference/datasets/")
    os.makedirs(new_data_path, exist_ok=True)
    new_data_file = os.path.join(new_data_path, "inputFile.jsonl")

    # Write to a local file
    try:
        mode = "a" if os.path.exists(new_data_file) else "w"
        with open(new_data_file, mode) as f:
            f.write(json.dumps(new_data) + "\n")
        logger.info("New feedback data written to local file.")
    except Exception as e:
        logger.error("Failed to write new feedback data to local file.", exc_info=True)

    # Upload to S3
    try:
        s3_client.upload_file(new_data_file, S3_BUCKET, f"{NEW_DATA_PATH}inputFile.jsonl")
        logger.info("New feedback data uploaded to S3.")
    except Exception as e:
        logger.error("Failed to upload new feedback data to S3.", exc_info=True)

# Calculate accuracy
def calculate_accuracy(predictions, ground_truth):
    return 1 if predictions == ground_truth else 0

# Measure CPU utilization
def get_cpu_utilization():
    return psutil.cpu_percent(interval=1)

def analyze_feedback(feedback):
    logger.info("Starting inference for new feedback.")
    pyRAPL.setup()
    rapl = pyRAPL.Measurement(label="inference")
    rapl.begin()
    try:

        # Tokenize the input text
        inputs = tokenizer(feedback.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        logger.info("Tokenization complete.")

        # Predict sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()
            sentiment = sentiment_labels[predictions]
        logger.info(f"Prediction complete. Sentiment: {sentiment}")

        # Feedback scoring based on stars
        stars_weight = feedback.stars / 5
        feedback_score = predictions + stars_weight
        logger.info(f"feedback score: {feedback_score}")

        # Accuracy
        ground_truth = 1 if feedback.stars >= 3 else 0
        accuracy = calculate_accuracy(predictions, ground_truth)

        # Additional metrics
        cpu_utilization = get_cpu_utilization()

        # Interpret overall sentiment
        if feedback_score <= 1:
            overall_sentiment = "Disappointed"
        elif feedback_score <= 2:
            overall_sentiment = "Angry"
        elif feedback_score <= 3:
            overall_sentiment = "Neutral"
        elif feedback_score <= 4:
            overall_sentiment = "Satisfied"
        else:
            overall_sentiment = "Happy"

        rapl.end()
        power_metrics = rapl.result.pkg
        power_consumption = power_metrics
        logger.info(f"Inference complete. Overall Sentiment: {overall_sentiment}")
        return sentiment, feedback_score, overall_sentiment, accuracy, cpu_utilization, power_consumption

    except Exception as e:
        logger.error("Error during inference.", exc_info=True)
        raise e

# API endpoint
@app.post("/feedback/analyse", response_model=FeedbackResponse)
def analyze(feedback: FeedbackRequest):
    if feedback.stars < 1 or feedback.stars > 5:
        logger.warning("Invalid stars value received.")
        raise HTTPException(status_code=400, detail="Stars must be between 1 and 5")

    # Log new data for retraining
    try:
        log_new_data_to_s3(feedback)
    except Exception as e:
        logger.error("Failed to log new feedback data.", exc_info=True)

    # Perform inference
    try:
        sentiment, feedback_score, overall_sentiment, accuracy, cpu_utilization, power_consumption = analyze_feedback(
            feedback)
        return FeedbackResponse(
            sentiment=overall_sentiment,
            feedback_score=feedback_score,
            accuracy=accuracy,
            cpu_utilization=cpu_utilization,
            power_consumption=power_consumption
        )
    except Exception as e:
        logger.error("Failed to process feedback.", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during inference.")