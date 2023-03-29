from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class ClassifyIntentExtractEntitiesV2Dto(BaseModel):
    session_id: str
    personality: str
    user_input: str
    intents: list[Intent]
    entities: list[Entity]


@app.post("/v3/extract_intent_and_entities")
async def classify_v3(request: ClassifyIntentExtractEntitiesV2Dto):
    classify_response = await classify_intent_extract_entities(
        user_input=jsonable_encoder(request.user_input),
        intents=jsonable_encoder(request.intents),
        entities=jsonable_encoder(request.entities),
    )
    print(classify_response)
