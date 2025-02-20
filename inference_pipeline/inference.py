from fastapi import FastAPI
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import boto3
import os
import logging
import queue
from FeedbackAnalysis import FeedbackAnalysis


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

app = FastAPI()
feedback_queue = queue.Queue()
os.makedirs(new_data_path_local, exist_ok=True)

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
                except Exception as e1:
                    logger.error(f"Error downloading {file_name} from S3: {e1}")
                    raise e1
    except Exception as e2:
        logger.error(f"Error listing files from S3: {e2}")
        raise e2
    logger.info("Model download complete.")
    mb_model = MobileBertForSequenceClassification.from_pretrained(f"{local_model_dir}")
    mb_tokenizer = MobileBertTokenizer.from_pretrained(f"{local_model_dir}")
    return mb_model, mb_tokenizer

try:
    model, tokenizer = download_model_from_s3()
    device = "cpu"
    model = model.to(device)
    feedback_analysis = FeedbackAnalysis(app=app, feedback_queue=feedback_queue, new_data_file_local=new_data_file_local, logger=logger, model=model, tokenizer=tokenizer, s3_client=s3_client, s3_bucket=S3_BUCKET, new_data_path=NEW_DATA_PATH, device=device)
except Exception as e:
    logger.critical("Failed to load model. Service cannot start.", exc_info=True)
    raise RuntimeError("Model initialization failed.") from e