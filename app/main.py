from fastapi import FastAPI
from pydantic import BaseModel
import json
import openai
from fastapi.encoders import jsonable_encoder
from endpoints.v3 import (
    extract_entities_external_prompt,
)

from tools.sentiment import sentiment_emotion_hate
from langchain.prompts import load_prompt

# from tools.autoforms import out_of_scope, simple, multi_call, cleaned_form
from tools.classification import (
    classify_intent_extract_entities,
    classify_intent_external_prompt,
    classify_entities_external_prompt,
)
import requests

from tools.kb import (
    ask_retrieval,
    ingest,
    ask_conversational,
    conversational_ingest,
)
from tools.moderation import get_moderation_intent_entities

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
sentiment_url = os.getenv("SENTIMENT_URL")

app = FastAPI()


@app.get("/version")
async def version():
    return {"message": "2023-03-29  v0.1.8 endpoints hasta v3"}


class SentimentDto(BaseModel):
    user_input: str


@app.post("/v2/sentiment")
async def sentiment_full(request: SentimentDto):
    response = await sentiment_emotion_hate(request.user_input)
    return response


class Intent(BaseModel):
    name: str
    description: str


class Entity(BaseModel):
    name: str
    description: str


class ExtractIntentAndEntitiesDto(BaseModel):
    user_input: str
    intents: list[Intent]
    entities: list[Entity]


class IntentEntitiesDto(BaseModel):
    intent: dict
    entities: list[dict]


class KbResource(BaseModel):
    url: str
    type: str


class KbIngestDto(BaseModel):
    resources: list[KbResource]


class ExtractEntitiesExternalPromptDto(BaseModel):
    user_input: str
    prompt: str
    session_id: str


class ExtractEntitiesExternalPromptDto(BaseModel):
    user_input: str
    prompt: str
    session_id: str


@app.post("/v3/extract_entities_external_prompt")
async def classify_v3_external_prompt(
    request: ExtractEntitiesExternalPromptDto,
):
    entities_response = await classify_entities_external_prompt(
        user_input=jsonable_encoder(request.user_input),
        prompt=jsonable_encoder(request.prompt),
    )
    print(entities_response)
    # return {"intent": {"name": intent_response.content, "confidence": 1.0}}
    return entities_response


class ExtractIntentExternalPromptDto(BaseModel):
    user_input: str
    prompt: str
    session_id: str


@app.post("/v3/extract_intent_external_prompt")
async def classify_v3_external_prompt(request: ExtractIntentExternalPromptDto):
    # moderation = get_moderation_intent_entities(
    #     request.user_input, request.session_id
    # )
    # flagged = moderation["flagged"]
    # if flagged:
    #     return moderation

    intent_response = await classify_intent_external_prompt(
        user_input=jsonable_encoder(request.user_input),
        prompt=jsonable_encoder(request.prompt),
    )
    print(intent_response)
    # return {"intent": {"name": intent_response.content, "confidence": 1.0}}
    return intent_response


class KbResource(BaseModel):
    url: str
    type: str


class KbIngestDto(BaseModel):
    resources: list[KbResource]


@app.post("/v2/kb/ingest")
async def kb_ingest(request: KbIngestDto):
    resources = jsonable_encoder(request.resources)
    response = await conversational_ingest(resources)
    return response


class KbAskDto(BaseModel):
    session_id: str
    personality: str
    user_input: str


@app.post("/v2/kb/ask")
async def kb_ask(request: KbAskDto):
    # moderation = get_moderation_intent_entities(
    #     request.user_input, request.session_id
    # )
    # flagged = moderation["flagged"]
    # if flagged:
    #     return moderation
    kb_response = ask_conversational(
        user_input=request.user_input,
        session_id=request.session_id,
        personality=request.personality,
    )
    seh = await sentiment_emotion_hate(request.user_input)
    response = {
        "session_id": request.session_id,
        "user_input": request.user_input,
        "sentiment": seh["sentiment"],
        "emotion": seh["emotion"],
        "hate_speech": seh["hate_speech"],
        **kb_response,
    }
    return response
