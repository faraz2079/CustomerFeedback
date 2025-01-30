from fastapi import FastAPI, HTTPException, BackgroundTasks
from feedback_request_model import FeedbackRequest
from feedback_response_model import FeedbackResponse
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from hwcounter import Timer
import boto3
import os
import psutil
import json
import logging
import fcntl
import time
import threading
import queue

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
# AWS S3 Configuration
S3_BUCKET = "customerfeedbackmlbucket"
MODEL_PATH = "models/"
NEW_DATA_PATH = "datasets/"
local_model_dir = os.path.expanduser("~/s3/inference/models/")
new_data_path_local = os.path.expanduser("~/s3/inference/datasets/")
new_data_file_local = os.path.join(new_data_path_local, "inputFile.jsonl")
s3_client = boto3.client('s3', region_name='eu-central-1')

system_metrics = {"cpu": 0.0, "ram": 0.0}

app = FastAPI()
feedback_queue = queue.Queue()

# Load the model and tokenizer from S3
def download_model_from_s3():
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
                local_file_path = os.path.join(f"{local_model_dir}", f"{file_name}")
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
    model = MobileBertForSequenceClassification.from_pretrained(f"{local_model_dir}")
    tokenizer = MobileBertTokenizer.from_pretrained(f"{local_model_dir}")
    return model, tokenizer

# Log new data to S3
def create_new_input_file(feedback):
    new_data = {
        "text": feedback.text,
        "stars": feedback.stars
    }
    feedback_queue.put(json.dumps(new_data) + "\n")

def batch_write_feedback():
    while True:
        try:
            if not feedback_queue.empty():
                with open(new_data_file_local, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    while not feedback_queue.empty():
                        f.write(feedback_queue.get())
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error("Batch write failed.", exc_info=True)
        time.sleep(5)

os.makedirs(new_data_path_local, exist_ok=True)
threading.Thread(target=batch_write_feedback, daemon=True).start()

def monitor_system():
    while True:
        system_metrics["cpu"] = psutil.cpu_percent(interval=1)
        system_metrics["ram"] = psutil.virtual_memory().used / 1e9

threading.Thread(target=monitor_system, daemon=True).start()

# Calculate accuracy
def calculate_accuracy(predictions):
    return predictions / 4.0

def analyze_feedback(feedback):
    logger.info("Starting inference for new feedback.")
    try:
        tokens = tokenizer.tokenize(feedback.text.lower())
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        inputs = torch.tensor([input_ids]).to(device)
        logger.info("Tokenization complete.")

        # Predict sentiment
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()
            sentiment = sentiment_labels[predictions]
        logger.info(f"Prediction complete. Prediction: {predictions} and Sentiment: {sentiment}")

        # Feedback scoring based on stars
        stars_weight = feedback.stars / 5
        feedback_score = predictions + stars_weight
        logger.info(f"feedback score: {feedback_score}")

        # Accuracy
        accuracy = calculate_accuracy(predictions)

        # Interpret overall sentiment
        if feedback_score <= 1:
            overall_sentiment = "Angry"
        elif feedback_score <= 2:
            overall_sentiment = "Disappointed"
        elif feedback_score <= 3:
            overall_sentiment = "Neutral"
        elif feedback_score <= 4:
            overall_sentiment = "Satisfied"
        else:
            overall_sentiment = "Happy"

        # Additional metrics
        cpu_utilization = system_metrics["cpu"]
        ram_usage = system_metrics["ram"]
        logger.info(f"Inference complete. Overall Sentiment: {overall_sentiment}")
        return sentiment, feedback_score, overall_sentiment, accuracy, cpu_utilization, ram_usage

    except Exception as e:
        logger.error("Error during inference.", exc_info=True)
        raise e

# API endpoint
@app.post("/feedback/analyse", response_model=FeedbackResponse)
def analyze(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    if feedback.stars < 1 or feedback.stars > 5:
        logger.warning("Invalid stars value received.")
        raise HTTPException(status_code=400, detail="Stars must be between 1 and 5")

    background_tasks.add_task(create_new_input_file, feedback)

    # Perform inference and send response
    try:
        with Timer() as t:
            sentiment, feedback_score, overall_sentiment, accuracy, cpu_utilization, ram_usage = analyze_feedback(
            feedback)
        elapsed_cycles = t.cycles
        end = time.perf_counter()
        execution_time = end - start
        logger.info(f"Final Analysis: " +
                    f"Sentiment: {sentiment}" +
                    f"Overall sentiment: {overall_sentiment}" +
                    f"Feedback score: {feedback_score}" +
                    f"Accuracy: {accuracy}" +
                    f"CPU utilization: {cpu_utilization}" +
                    f"RAM usage: {ram_usage}" +
                    f"CPU cycle: {elapsed_cycles}" +
                    f"Inference time: {execution_time}")
        return FeedbackResponse(
            sentiment=overall_sentiment,
            feedback_score=round(feedback_score, 2),
            accuracy=round(accuracy, 2),
            cpu_utilization=round(cpu_utilization, 2),
            cpu_cycles=elapsed_cycles,
            ram_usage=round(ram_usage, 2),
            inference_time=round(execution_time, 2)
        )
    except Exception as e:
        logger.error("Failed to process feedback.", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during inference.")

@app.get("/uploadInputFile")
def upload_new_datafile():
    # Upload to S3
    try:
        s3_client.upload_file(new_data_file_local, S3_BUCKET, f"{NEW_DATA_PATH}inputFile.jsonl")
        logger.info("New feedback data uploaded to S3.")
    except Exception as e:
        logger.error("Failed to upload new feedback data to S3.", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload file.")


try:
    model, tokenizer = download_model_from_s3()
    device = "cpu"
    model = model.to(device)
except Exception as e:
    logger.critical("Failed to load model. Service cannot start.", exc_info=True)
    raise RuntimeError("Model initialization failed.") from e

# Sentiment labels
sentiment_labels = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}