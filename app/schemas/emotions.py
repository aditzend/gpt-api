from pydantic import BaseModel, Field


class ExtractEmotionsDto(BaseModel):
    user_input: str
    emotions: list
