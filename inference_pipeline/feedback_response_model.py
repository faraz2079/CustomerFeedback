from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    sentiment: str
    feedback_score: float
    accuracy: float
    inference_time: float
    pod_name: str