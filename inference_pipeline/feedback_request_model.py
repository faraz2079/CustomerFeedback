from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    text: str
    stars: int