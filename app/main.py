from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
import time
import openai
from fastapi.encoders import jsonable_encoder
from faiss_kb import faiss_retrieval, faiss_ask_retrieval, faiss_ingest
from tools.kbdeeplake import deeplake_ask_retrieval, deeplake_retrieval_ingest

from tools.sentiment import sentiment_emotion_hate
from langchain.prompts import load_prompt

# from tools.autoforms import out_of_scope, simple, multi_call, cleaned_form
from tools.classification import (
    classify_intent_extract_entities,
    classify_intent_external_prompt,
    classify_entities_external_prompt,
    bulk_intent_classifier,
    bulk_classifier,
    bulk_entity_classifier,
)
import requests

from schemas import KbResource, KbIngestDto

# from tools.kbdeeplakecohere import (
#     deeplake_retrieval_ingest,
#     deeplake_ask_retrieval,
# )


# from tools.kbweaviate import (
#     weaviate_retrieval_ingest,
#     weaviate_ask_retrieval,
# )

# from tools.kb import (
#     ask_retrieval,
#     retrieval_ingest,
#     ask_conversational,
#     conversational_ingest,
#     delete_all_indexes,
#     make_post_prompts_from_copy,
#     make_post_prompts_from_url,
# )
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
        "date": "2023-04-03",
        "branch": "env-configs",
        "version": "0.1.16",
        "comments": "endpoints hasta v3",
    }
    print(version)
    return version


# *******************************  V3 *********************************
@app.post("/v3/kb/ingest")
async def faiss_ingestor(request: KbIngestDto):
    resources = jsonable_encoder(request.resources)
    response = await faiss_ingest(resources)
    return response


# Retrieval
class KbRetrievalDto(BaseModel):
    user_input: str
    index: str = "main"
    ranking_size: int = 3


@app.post("/v3/kb/retrieve")
async def faiss_retrieve(request: KbRetrievalDto):
    response = await faiss_retrieval(
        user_input=jsonable_encoder(request.user_input),
        index=jsonable_encoder(request.index),
        ranking_size=jsonable_encoder(request.ranking_size),
    )
    return response


# @app.post("/v3/kb/ask")
# async def faiss_ask(request: KbAskDto):
#     user_input = jsonable_encoder(request.user_input)
#     index = jsonable_encoder(request.index)
#     company_name = jsonable_encoder(request.company_name)
#     default_response = jsonable_encoder(request.default_response)
#     language = jsonable_encoder(request.language)
#     session_id = jsonable_encoder(request.session_id)
#     personality = jsonable_encoder(request.personality)
#     emoji_level = jsonable_encoder(request.emoji_level)

#     kb_response = await faiss_ask_retrieval(
#         user_input=user_input,
#         session_id=session_id,
#         index=index,
#         personality=personality,
#         company_name=company_name,
#         default_response=default_response,
#         language=language,
#         emoji_level=emoji_level,
#     )
#     response = {
#         "session_id": "dev",
#         "user_input": user_input,
#         **kb_response,
#     }
#     return response


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


class FlexKbResource(BaseModel):
    url: str
    type: str
    index: str = Field(default="main")


class FlexKbIngestDto(BaseModel):
    resources: list[FlexKbResource]


@app.post("/v2/kb/ingest")
async def faiss_ingestor_flex(request: FlexKbIngestDto):
    resources = jsonable_encoder(request.resources)
    response = await faiss_ingest(resources)
    return response


@app.post("/v3/deeplake/ingest")
async def deeplake_ingest(request: KbIngestDto):
    resources = jsonable_encoder(request.resources)
    response = await deeplake_retrieval_ingest(resources)
    return response


class KbAskDto(BaseModel):
    session_id: str = "test"
    personality: str = "amable"
    user_input: str
    language: str = "español de Argentina"
    company_name: str = "TestCo"
    default_response: str = "Lo siento, no puedo responder eso"
    index: str = "main"
    emoji_level: int = 0


@app.post("/v3/deeplake/ask")
async def deeplake_ask(request: KbAskDto):
    kb_response = {}
    kb_response = await deeplake_ask_retrieval(
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


class FlexKbAskDto(BaseModel):
    session_id: str = "test"
    personality: str = "amable"
    user_input: str
    # language: str = "español de Argentina"
    # company_name: str = "TestCo"
    # default_response: str = "Lo siento, no puedo responder eso"
    # index: str = "main"
    # emoji_level: int = 0


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
