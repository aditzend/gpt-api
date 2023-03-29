from fastapi import FastAPI
from pydantic import BaseModel
from tools.sentiment import sentiment_emotion_hate

app = FastAPI()


class SentimentDto(BaseModel):
    user_input: str


@app.post("/v2/sentiment")
async def sentiment_full(request: SentimentDto):
    response = await sentiment_emotion_hate(request.user_input)
    return response
