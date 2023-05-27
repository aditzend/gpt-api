from pydantic import BaseModel, Field


class SentimentDto(BaseModel):
    user_input: str
