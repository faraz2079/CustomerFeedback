import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# AWS Configuration
S3_BUCKET = "sagemaker-eu-central-1-528757790533"  # Replace with your bucket name
DATA_PATH = "customerfeedbackml/datasets/inputFile.json"  # Update with dataset S3 location
SCRIPT_PATH = "customerfeedbackml/scripts/trainer_aws_script.py"
MODEL_OUTPUT_PATH = "customerfeedbackml/models/mobilebert_sentiment_model/"
ROLE_ARN = "arn:aws:iam::528757790533:role/service-role/AmazonSageMaker-ExecutionRole-20250110T214828"  # Replace with your IAM role ARN

# S3 client
s3_client = boto3.client("s3")

#def upload_to_s3(local_path, s3_key):
    #logger.info(f"Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}...")
    #s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    #logger.info(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")

def create_huggingface_estimator(is_initial_training):
    huggingface_estimator = HuggingFace(
        entry_point="trainer_aws_script.py",
        source_dir="s3://sagemaker-eu-central-1-528757790533/customerfeedbackml/scripts",
        role=ROLE_ARN,
        instance_type="ml.p3.2xlarge",  # Use GPU-enabled instance
        instance_count=1,
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        hyperparameters={
            "epochs": 3,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "learning_rate": 5e-5,
            "max_seq_length": 128,
            "model_name": "google/mobilebert-uncased",
            "is_initial_training": int(is_initial_training)  # Pass as integer (1 for True, 0 for False)
        },
        output_path=f"s3://{S3_BUCKET}/{MODEL_OUTPUT_PATH}",
    )
    return huggingface_estimator

def main():
    # Upload script and data to S3
    #upload_to_s3("inputFile.json", DATA_PATH)
    #upload_to_s3("train_aws_script.py", SCRIPT_PATH)

    # Define whether it’s initial or incremental training
    is_initial_training = True  # Set this flag as required

    # Create and start training job
    logger.info("Starting SageMaker training job...")
    estimator = create_huggingface_estimator(is_initial_training)
    #estimator.fit({"train": f"s3://{S3_BUCKET}/{DATA_PATH}"})
    estimator.fit()
    logger.info(f"Training complete. Model saved to s3://{S3_BUCKET}/{MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
