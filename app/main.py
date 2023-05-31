from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import openai
import os
from fastapi.encoders import jsonable_encoder
from tools.faiss_kb import (
    faiss_retrieval,
    faiss_ask_retrieval,
    faiss_ingest,
    # faiss_ask_streaming,
)
from tools.sentiment import sentiment_emotion_hate
from tools.classification import (
    bulk_intent_classifier,
    bulk_classifier,
    bulk_entity_classifier,
)
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn")

if os.getenv("LOG_LEVEL") == "DEBUG":
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG

# # Add a console handler to display logs in the console
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)  # Set the log level for the handler
# logger.addHandler(handler)

from schemas.kb import (
    FlexKbAskDto,
    KbIngestDto,
    KbRetrievalDto,
    FlexKbIngestDto,
    KbAskDto,
)
from schemas.emotions import ExtractEmotionsDto
from schemas.sentiment import SentimentDto

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
COMPANY_NAME = os.getenv("COMPANY_NAME")

app = FastAPI()


@app.get("/version")
async def version():
    version = {
        "date": "2023-05-27",
        "branch": "028",
        "version": "0.2.8",
        "comments": "html loading and main db duplication",
    }
    print(version)
    return version


# *******************************  V3 *********************************


# Ingestion
@app.post("/v3/kb/ingest")
async def faiss_ingestor(request: KbIngestDto):
    logger.info("faiss ingestor")
    resources = jsonable_encoder(request.resources)
    response = await faiss_ingest(resources)
    return response


# V3 QA
@app.post("/v3/kb/ask")
async def faiss_ask_index(request: KbAskDto):
    user_input = jsonable_encoder(request.user_input)
    index = jsonable_encoder(request.index)
    company_name = COMPANY_NAME or jsonable_encoder(request.company_name)
    default_response = jsonable_encoder(request.default_response)
    language = jsonable_encoder(request.language)
    session_id = jsonable_encoder(request.session_id)
    personality = jsonable_encoder(request.personality)
    emoji_level = jsonable_encoder(request.emoji_level)

    # kb_response = await faiss_ask_streaming(
    #     user_input=user_input,
    #     session_id=session_id,
    #     index=index,
    #     personality=personality,
    #     company_name=company_name,
    #     default_response=default_response,
    #     language=language,
    #     emoji_level=emoji_level,
    # )
    # kb_response = await faiss_ask_streaming()
    return "Not available"


# Retrieval
@app.post("/v3/kb/retrieve")
async def faiss_retrieve(request: KbRetrievalDto):
    response = await faiss_retrieval(
        user_input=jsonable_encoder(request.user_input),
        index=jsonable_encoder(request.index) or "main",
        ranking_size=jsonable_encoder(request.ranking_size),
    )
    return response


# Emotion classifiers
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


# *******************************  V2 *********************************


# Sentiment
@app.post("/v2/sentiment")
async def sentiment_full(request: SentimentDto):
    response = await sentiment_emotion_hate(request.user_input)
    return response


# V2 Ingestor
@app.post("/v2/kb/ingest")
async def faiss_ingestor_flex(request: FlexKbIngestDto):
    logging.info("[[ main faiss_ingestor_flex ]] /v2/kb/ingest REQUEST")
    resources = jsonable_encoder(request.resources)
    response = await faiss_ingest(resources)
    return response


# V2 Ask
@app.post("/v2/kb/ask")
async def faiss_ask_flex(request: FlexKbAskDto):
    user_input = jsonable_encoder(request.user_input)
    index = "main"
    company_name = COMPANY_NAME
    default_response = "Lo siento, no puedo responder eso."
    language = "español de Argentina"
    session_id = jsonable_encoder(request.session_id)
    personality = jsonable_encoder(request.personality)
    emoji_level = 4

    kb_response = await faiss_ask_retrieval(
        user_input=user_input,
        session_id=session_id,
        index=index,
        personality=personality,
        company_name=company_name,
        default_response=default_response,
        language=language,
        emoji_level=emoji_level,
    )
    response = {
        "session_id": session_id,
        "user_input": user_input,
        **kb_response,
    }
    return response
    response = {
        "session_id": request.session_id,
        "user_input": request.user_input,
        "sentiment": seh["sentiment"],
        "emotion": seh["emotion"],
        "hate_speech": seh["hate_speech"],
        **kb_response,
    }
    return response


# ************************ SYSTEM ******************************


class SystemPromptDto(BaseModel):
    prompt: str


@app.post("/v3/system/completion")
async def system_completion(request: SystemPromptDto):
    prompt = jsonable_encoder(request.prompt)
    response = await system_completion_v1_turbo_t0(content=prompt)
    response = {"content": response}
    return response


# ********************** DEV *******************


class BulkIntent(BaseModel):
    n: str
    d: str
    k: int


class BulkEntity(BaseModel):
    n: str
    d: str


class BulkClassificationDto(BaseModel):
    user_input: str
    intents: list[BulkIntent]
    entities: list[BulkEntity]


@app.post("/dev/bulk_intent_classifier")
async def classify_intent_bulk(request: BulkClassificationDto):
    user_input = jsonable_encoder(request.user_input)
    intents = jsonable_encoder(request.intents)
    start_time = time.time()
    response = await bulk_intent_classifier(
        intents=intents,
        user_input=user_input,
    )
    end_time = time.time()
    intent_classification_time = end_time - start_time
    return {
        "user_input": user_input,
        "intent": response,
        "intent_classification_time": intent_classification_time,
    }


@app.post("/dev/bulk_entity_classifier")
async def classifybulk(request: BulkClassificationDto):
    user_input = jsonable_encoder(request.user_input)
    # intents = jsonable_encoder(request.intents)
    entities = jsonable_encoder(request.entities)
    start_time = time.time()
    response = await bulk_entity_classifier(
        user_input=user_input, entities=entities
    )
    entities = json.loads(response)
    end_time = time.time()
    entity_extraction_time = end_time - start_time
    return {
        "user_input": user_input,
        **entities,
        "entity_extraction_time": entity_extraction_time,
    }


@app.post("/dev/bulk_classifier")
async def classify_all(request: BulkClassificationDto):
    user_input = jsonable_encoder(request.user_input)
    intents = jsonable_encoder(request.intents)
    start_time = time.time()
    response = await bulk_classifier(
        intents=intents,
        user_input=user_input,
    )
    end_time = time.time()
    intent_classification_time = end_time - start_time
    return {
        "user_input": user_input,
        "intent": response,
        "intent_classification_time": intent_classification_time,
    }
