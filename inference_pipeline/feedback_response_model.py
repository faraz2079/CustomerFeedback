from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    sentiment: str
    latency: float
    feedback_score: float
    accuracy: float
    cpu_utilization: float
    power_consumption: float