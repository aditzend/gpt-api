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

from tools.kbdeeplake import (
    v3_retrieval_ingest,
    v3_ask_retrieval,
)

from tools.kb import (
    ask_retrieval,
    retrieval_ingest,
    ask_conversational,
    conversational_ingest,
    delete_all_indexes,
    make_post_prompts_from_copy,
    make_post_prompts_from_url,
)
from tools.moderation import get_moderation_intent_entities

from tools.emotions import (
    classify_2_emotions_score_d,
    classify_1_emotion_score_d,
    classify_1_emotion_d,
)
from tools.openaidirect import system_completion_v1_turbo_t0

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
sentiment_url = os.getenv("SENTIMENT_URL")
kb_conversational = os.getenv("KB_CONVERSATIONAL")

app = FastAPI()


@app.get("/version")
async def version():
    version = {
        "date": "2023-04-03",
        "branch": "env-configs",
        "version": "0.1.16",
        "comments": "endpoints hasta v3",
    }
    print(version)
    return version


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
    if kb_conversational:
        response = await conversational_ingest(resources)
    else:
        response = await retrieval_ingest(resources)
    return response


@app.post("/v3/kb/ingest")
async def kb_ingest(request: KbIngestDto):
    resources = jsonable_encoder(request.resources)
    response = await v3_retrieval_ingest(resources)
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
    kb_response = {}
    if kb_conversational:
        kb_response = await ask_conversational(
            user_input=request.user_input,
            session_id=request.session_id,
            personality=request.personality,
        )
    else:
        kb_response = await ask_retrieval(
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


@app.post("/v3/kb/ask")
async def v3_kb_ask(request: KbAskDto):
    kb_response = {}
    kb_response = await v3_ask_retrieval(
        user_input=request.user_input,
        session_id=request.session_id,
        personality=request.personality,
    )
    response = {
        "session_id": request.session_id,
        "user_input": request.user_input,
        **kb_response,
    }
    return response


class ExtractEmotionsDto(BaseModel):
    user_input: str
    emotions: list


@app.post("/v3/emotions")
async def extract_emotions(request: ExtractEmotionsDto):
    user_input = jsonable_encoder(request.user_input)
    emotions = jsonable_encoder(request.emotions)
    response = await classify_2_emotions_score_d(
        emotions=emotions, user_input=user_input
    )
    return response


@app.post("/v3/emotions/2_score")
async def extract_2_emotions_score(request: ExtractEmotionsDto):
    user_input = jsonable_encoder(request.user_input)
    emotions = jsonable_encoder(request.emotions)
    response = await classify_2_emotions_score_d(
        emotions=emotions, user_input=user_input
    )
    return response


@app.post("/v3/emotions/1_score")
async def extract_1_emotion_score(request: ExtractEmotionsDto):
    user_input = jsonable_encoder(request.user_input)
    emotions = jsonable_encoder(request.emotions)
    response = await classify_1_emotion_score_d(
        emotions=emotions, user_input=user_input
    )
    return response


@app.post("/v3/emotions/1")
async def extract_1_emotion(request: ExtractEmotionsDto):
    user_input = jsonable_encoder(request.user_input)
    emotions = jsonable_encoder(request.emotions)
    response = await classify_1_emotion_d(
        emotions=emotions, user_input=user_input
    )
    return response


class SystemPromptDto(BaseModel):
    prompt: str


@app.post("/v3/system/completion")
async def system_completion(request: SystemPromptDto):
    prompt = jsonable_encoder(request.prompt)
    response = await system_completion_v1_turbo_t0(content=prompt)
    response = {"content": response}
    return response


@app.get("/v3/delete_all_kb_indexes")
async def delete_all_kb_indexes():
    response = delete_all_indexes()
    return response


class MakePromptDto(BaseModel):
    url: str
    post_example: str


@app.post("/v3/make_posts")
async def make_social_posts(request: MakePromptDto):
    url = jsonable_encoder(request.url)
    post_example = jsonable_encoder(request.post_example)
    response = make_post_prompts_from_url(url, post_example)
    return response


class MakePromptFromCopyDto(BaseModel):
    article_text: str


@app.post("/v3/make_posts_from_copy")
async def make_social_posts(request: MakePromptFromCopyDto):
    article_text = jsonable_encoder(request.article_text)
    response = make_post_prompts_from_copy(article_text)
    return response
