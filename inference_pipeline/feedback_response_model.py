from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    sentiment: str
    feedback_score: float
    accuracy: float
    cpu_utilization: float
    cpu_cycles: int
    ram_usage: float