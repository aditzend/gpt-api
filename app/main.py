from fastapi import FastAPI
from pydantic import BaseModel
import json
import openai
import os
from fastapi.encoders import jsonable_encoder
from langchain.prompts import load_prompt
from autoforms import out_of_scope
from classification import classify_intent_extract_entities
import requests

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
sentiment_url = os.getenv("SENTIMENT_URL")


def model_prediction(text, model):
    print("model called")
    response = openai.Completion.create(
        engine=model,
        prompt=text,
        max_tokens=1024,
        n=1,
        temperature=0.5,
        stop=None,
    )
    return response.choices[0].text


app = FastAPI()


class Intent(BaseModel):
    name: str
    description: str


class SlotToFill(BaseModel):
    name: str
    description: str


class SlotFilled(BaseModel):
    name: str
    value: str


class Entity(BaseModel):
    name: str
    description: str


class ExtractIntentAndEntitiesDto(BaseModel):
    user_input: str
    intents: list[Intent]
    entities: list[Entity]


class ClassifyIntentExtractEntitiesV2Dto(BaseModel):
    session_id: str
    personality: str
    user_input: str
    intents: list[Intent]
    entities: list[Entity]


class IntentEntitiesDto(BaseModel):
    intent: dict
    entities: list[dict]


class AutoformDto(BaseModel):
    session_id: str
    user_input: str
    slots_to_fill: list[SlotToFill]
    slots_filled: list[SlotFilled]
    personality: str


class SentimentDto(BaseModel):
    text: str


async def sentiment_emotion_hate(text):
    endpoint = sentiment_url + "/full/"
    response = requests.post(endpoint, json={"text": text})
    return response.json()


@app.post("/v2/sentiment")
async def sentiment_full(request: SentimentDto):
    response = await sentiment_emotion_hate(request.text)
    return response


@app.get("/version")
async def version():
    return {"message": "2: sentiment"}


@app.post("/v2/autoform")
async def autoformv2(body: AutoformDto):
    seh = await sentiment_emotion_hate(body.user_input)
    form_response = await out_of_scope(
        user_input=jsonable_encoder(body.user_input),
        slots_to_fill=jsonable_encoder(body.slots_to_fill),
        slots_filled=jsonable_encoder(body.slots_filled),
        personality=jsonable_encoder(body.personality),
    )
    response = {
        "session_id": body.session_id,
        "user_input": body.user_input,
        "sentiment": seh["sentiment"],
        "emotion": seh["emotion"],
        "hate_speech": seh["hate_speech"],
        **form_response,
    }
    return response


class ClassificationDto(BaseModel):
    confidence: float
    intent: str
    entities: list[Entity]


@app.post("/v2/extract_intent_and_entities")
async def classify_v2(request: ClassifyIntentExtractEntitiesV2Dto):
    moderation = openai.Moderation.create(input=request.user_input)
    flagged = moderation["results"][0]["flagged"]
    if flagged:
        return moderation["results"][0]

    classify_response: IntentEntitiesDto = (
        await classify_intent_extract_entities(
            user_input=jsonable_encoder(request.user_input),
            intents=jsonable_encoder(request.intents),
            entities=jsonable_encoder(request.entities),
        )
    )
    seh_response = await sentiment_emotion_hate(request.user_input)

    if classify_response:
        entities = []
        if "entities" in classify_response:
            entities = classify_response["entities"]
            if entities:
                for entity in entities:
                    entity["start"] = request.user_input.find(entity["text"])
                    entity["end"] = entity["start"] + len(entity["text"])
        response = {
            "session_id": request.session_id,
            "user_input": request.user_input,
            "sentiment": seh_response["sentiment"],
            "emotion": seh_response["emotion"],
            "hate_speech": seh_response["hate_speech"],
            "intent": classify_response["intent"],
            "entities": entities,
        }
    return response


@app.post("/v1/extract_intent_and_entities")
async def parse_text(request: ExtractIntentAndEntitiesDto):
    intents = jsonable_encoder(request.intents)
    entities = jsonable_encoder(request.entities)
    prompt = load_prompt("./prompts/classifier.json")

    hydrated = prompt.format(
        user_input=request.user_input,
        intents=intents,
        entities=entities,
    )
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": hydrated},
        ],
    )

    content = openai_response.choices[0].message.content
    if openai_response:
        p = json.loads(content)
        print(type(p))
        print(p)
        entities = []
        if "entities" in p:
            entities = p["entities"]
            if entities:
                for entity in entities:
                    entity["start"] = request.user_input.find(entity["text"])
                    entity["end"] = entity["start"] + len(entity["text"])
                    entity["confidence"] = 1.0
        response = {
            "user_input": request.user_input,
            "intent": {
                "name": p["intent"],
                "confidence": p["confidence"],
            },
            "entities": entities,
        }
    return response


# @app.post("/v1/autoform")
# async def autoform(request: AutoformDto):
#     slots_to_fill = jsonable_encoder(request.slots_to_fill)
#     slots_filled = jsonable_encoder(request.slots_filled)
#     hydrated = prompt.format(
#         user_input=request.user_input,
#         slots_to_fill=slots_to_fill,
#         slots_filled=slots_filled,
#     )
#     openai_response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": hydrated},
#         ],
#     )

#     content = openai_response.choices[0].message.content
#     if openai_response:
#         p = json.loads(content)
#         print(type(p))
#         print(p)
#         entities = []
#         if "entities" in p:
#             entities = p["entities"]
#             if entities:
#                 for entity in entities:
#                     entity["start"] = request.user_input.find(entity["text"])
#                     entity["end"] = entity["start"] + len(entity["text"])
#                     entity["confidence"] = 1.0
#         response = {
#             "user_input": request.user_input,
#             "intent": {
#                 "name": p["intent"],
#                 "confidence": p["confidence"],
#             },
#             "entities": entities,
#         }
#     return response
