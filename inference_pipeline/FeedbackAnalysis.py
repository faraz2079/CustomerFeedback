from feedback_request_model import FeedbackRequest
from feedback_response_model import FeedbackResponse
from fastapi import FastAPI, HTTPException
import threading
import time
import torch
import json

# Sentiment labels
sentiment_labels = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

class FeedbackAnalysis(threading.Thread):
    def __init__(self, app: FastAPI, feedback_queue, new_data_file_local, logger, model, tokenizer, s3_client, s3_bucket, new_data_path, device):
        super().__init__(daemon=True)
        self.app = app
        self.feedback_queue = feedback_queue
        self.new_data_file_local = new_data_file_local
        self.logger = logger
        self.model = model
        self.tokenizer = tokenizer
        self.s3_client = s3_client
        self.S3_BUCKET = s3_bucket
        self.NEW_DATA_PATH = new_data_path
        self.device = device
        self.running = True
        self.initialize_routes()

    def initialize_routes(self):
        @self.app.post("/feedback/analyse", response_model=FeedbackResponse)
        async def analyze(feedback:FeedbackRequest):
            return await self.analyze(feedback)

        @self.app.get("/uploadInputFile")
        async def upload_new_datafile():
            return await self.upload_new_datafile()

    def run(self):
        self.logger.info("FeedbackAnalysis thread started.")

    def stop(self):
        self.running = False
        self.join()

    # Log new data to S3
    def create_new_input_file(self, feedback):
        new_data = {
            "text": feedback.text,
            "stars": feedback.stars
        }
        self.feedback_queue.put(json.dumps(new_data) + "\n")

    def write_to_file(self):
        while True:
            try:
                if not self.feedback_queue.empty():
                    with open(self.new_data_file_local, "a") as f:
                        while not self.feedback_queue.empty():
                            feedback_data = self.feedback_queue.get()
                            f.write(feedback_data)
            except Exception:
                self.logger.error("write failed.", exc_info=True)

    # Calculate accuracy
    @staticmethod
    def calculate_accuracy(feedback_score):
        if 0 < feedback_score < 1.0:
            return feedback_score
        elif 1.0 <= feedback_score < 5.0:
            return feedback_score / 5.0
        else:
            return 1.0

    async def analyze_feedback(self,feedback):
        self.logger.info("Starting inference for new feedback.")
        try:
            tokens = self.tokenizer.tokenize(feedback.text.lower())
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor([input_ids]).to(self.device)
            self.logger.info("Tokenization complete.")

            # Predict sentiment
            with torch.no_grad():
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs.logits, dim=1).item()
                sentiment = sentiment_labels[predictions]
            self.logger.info(f"Prediction complete. Prediction: {predictions} and Sentiment: {sentiment}")

            # Feedback scoring based on stars
            stars_weight = feedback.stars / 5
            feedback_score = predictions + stars_weight
            self.logger.info(f"feedback score: {feedback_score}")

            # Accuracy
            accuracy = FeedbackAnalysis.calculate_accuracy(feedback_score)

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

            self.logger.info(f"Inference complete. Overall Sentiment: {overall_sentiment}")
            return sentiment, feedback_score, overall_sentiment, accuracy

        except Exception as e:
            self.logger.error("Error during inference.", exc_info=True)
            raise e

    async def analyze(self, feedback):
        start = time.perf_counter()
        if feedback.stars < 1 or feedback.stars > 5:
            self.logger.warning("Invalid stars value received.")
            raise HTTPException(status_code=400, detail="Stars must be between 1 and 5")

        self.create_new_input_file(feedback)

        # Perform inference and send response
        try:
            sentiment, feedback_score, overall_sentiment, accuracy = await self.analyze_feedback(
                feedback)
            end = time.perf_counter()
            execution_time = end - start
            self.logger.info(f"Final Analysis: " +
                        f"Sentiment: {sentiment} " +
                        f"Overall sentiment: {overall_sentiment} " +
                        f"Feedback score: {feedback_score} " +
                        f"Accuracy: {accuracy} " +
                        f"Inference time: {execution_time} ")
            return FeedbackResponse(
                sentiment=overall_sentiment,
                feedback_score=round(feedback_score, 2),
                accuracy=round(accuracy, 2),
                inference_time=round(execution_time, 2)
            )
        except Exception:
            self.logger.error("Failed to process feedback.", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred during inference.")

    def upload_new_datafile(self):
        # Upload to S3
        try:
            self.write_to_file()
            self.s3_client.upload_file(self.new_data_file_local, self.S3_BUCKET, f"{self.NEW_DATA_PATH}inputFile.jsonl")
            self.logger.info("New feedback data uploaded to S3.")
        except Exception:
            self.logger.error("Failed to upload new feedback data to S3.", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to upload file.")