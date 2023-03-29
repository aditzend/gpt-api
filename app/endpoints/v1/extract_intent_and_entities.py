import openai
from fastapi.encoders import jsonable_encoder
import json
from langchain.prompts import load_prompt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


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
