from fastapi import FastAPI
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import os
import logging
from feedback_analysis import FeedbackAnalysis

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

# Local model and data paths (inside Docker or local filesystem)
local_model_dir = "models"
new_data_path_local = "datasets"
new_data_file_local = os.path.join(new_data_path_local, "inputFile.jsonl")

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer from local directory
try:
    logger.info("Loading model and tokenizer from local directory...")
    model = MobileBertForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = MobileBertTokenizer.from_pretrained(local_model_dir)
    device = "cpu"
    model = model.to(device)

    feedback_analysis = FeedbackAnalysis(
        app=app,
        new_data_file_local=new_data_file_local,
        logger=logger,
        model=model,
        tokenizer=tokenizer,
        s3_client=None,
        s3_bucket=None,
        new_data_path=new_data_path_local,
        device=device
    )
    logger.info("Model and FeedbackAnalysis initialized successfully.")
except Exception as e:
    logger.critical("Failed to load model. Service cannot start.", exc_info=True)
    raise RuntimeError("Model initialization failed.")
