from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import MobileBertTokenizer, TFAutoModelForSequenceClassification
import boto3
import os
import time
import psutil
import pyrapl
from datetime import datetime
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
S3_BUCKET = "your-s3-bucket-name"
MODEL_PATH = "models/latest/"
NEW_DATA_PATH = "new_data/"

s3_client = boto3.client("s3")



class FeedbackRequest(BaseModel):
    text: str
    stars: int


class FeedbackResponse(BaseModel):
    sentiment: str
    latency: float
    feedback_score: float
    accuracy: float
    cpu_utilization: float
    power_consumption: float


# Load the model and tokenizer from S3
def download_model_from_s3():
    local_model_dir = "./model_dir"
    os.makedirs(local_model_dir, exist_ok=True)

    logger.info("Downloading model files from S3...")
    for file_name in ["tf_model.h5", "config.json"]:
        try:
            s3_client.download_file(S3_BUCKET, f"{MODEL_PATH}{file_name}", os.path.join(local_model_dir, file_name))
            logger.info(f"Successfully downloaded {file_name} from S3.")
        except Exception as e:
            logger.error(f"Error downloading {file_name} from S3: {e}")
            raise e

    logger.info("Model download complete.")
    model = TFAutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    return model, tokenizer


try:
    model, tokenizer = download_model_from_s3()
except Exception as e:
    logger.critical("Failed to load model. Service cannot start.", exc_info=True)
    raise RuntimeError("Model initialization failed.") from e

# Define sentiment labels
sentiment_labels = {0: "Negative", 1: "Positive"}


# Log new data to S3
def log_new_data_to_s3(feedback):
    new_data = {
        "text": feedback.text,
        "stars": feedback.stars,
        "timestamp": datetime.now().isoformat()
    }
    new_data_path = f"/tmp/new_feedback.jsonl"

    # Write to a local file
    try:
        mode = "a" if os.path.exists(new_data_path) else "w"
        with open(new_data_path, mode) as f:
            f.write(json.dumps(new_data) + "\n")
        logger.info("New feedback data written to local file.")
    except Exception as e:
        logger.error("Failed to write new feedback data to local file.", exc_info=True)

    # Upload to S3
    try:
        s3_client.upload_file(new_data_path, S3_BUCKET, f"{NEW_DATA_PATH}new_feedback.jsonl")
        logger.info("New feedback data uploaded to S3.")
    except Exception as e:
        logger.error("Failed to upload new feedback data to S3.", exc_info=True)


# Calculate accuracy
def calculate_accuracy(predictions, actual_label):
    return 1.0 if predictions == actual_label else 0.0


# Measure CPU utilization
def get_cpu_utilization():
    return psutil.cpu_percent(interval=1)


# Measure power consumption
def get_power_consumption():
    rapl = pyrapl.Measurement()
    power_metrics = rapl.measure()
    return power_metrics["package-0"]["energy (J)"]


# Inference logic
def analyze_feedback(feedback):
    logger.info("Starting inference for new feedback.")
    start_time = time.time()

    try:
        # Tokenize the input text
        inputs = tokenizer(feedback.text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logger.debug("Tokenization complete.")

        # Predict sentiment
        predictions = model(inputs)[0]
        predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
        sentiment = sentiment_labels[predicted_label]
        logger.debug(f"Prediction complete. Sentiment: {sentiment}")

        # Feedback scoring based on stars
        stars_weight = feedback.stars / 5
        feedback_score = predicted_label * stars_weight

        # End latency measurement
        end_time = time.time()
        latency = end_time - start_time

        # Accuracy
        accuracy = calculate_accuracy(predicted_label, 1 if feedback.stars >= 3 else 0)

        # Additional metrics
        cpu_utilization = get_cpu_utilization()
        power_consumption = get_power_consumption()

        # Interpret overall sentiment
        if feedback_score <= 1.5:
            overall_sentiment = "Disappointed"
        elif feedback_score <= 2.5:
            overall_sentiment = "Angry"
        elif feedback_score <= 3.5:
            overall_sentiment = "Neutral"
        elif feedback_score <= 4.5:
            overall_sentiment = "Satisfied"
        else:
            overall_sentiment = "Happy"

        logger.info(f"Inference complete. Overall Sentiment: {overall_sentiment}")
        return sentiment, feedback_score, latency, overall_sentiment, accuracy, cpu_utilization, power_consumption

    except Exception as e:
        logger.error("Error during inference.", exc_info=True)
        raise e


# API endpoint
@app.post("/feedback/analyze", response_model=FeedbackResponse)
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
        sentiment, feedback_score, latency, overall_sentiment, accuracy, cpu_utilization, power_consumption = analyze_feedback(
            feedback)
        return FeedbackResponse(
            sentiment=overall_sentiment,
            latency=latency,
            feedback_score=feedback_score,
            accuracy=accuracy,
            cpu_utilization=cpu_utilization,
            power_consumption=power_consumption
        )
    except Exception as e:
        logger.error("Failed to process feedback.", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during inference.")
