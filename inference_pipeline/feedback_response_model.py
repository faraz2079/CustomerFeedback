from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    sentiment: str
    feedback_score: float
    accuracy: float
    cpu_utilization: float
    power_consumption: float