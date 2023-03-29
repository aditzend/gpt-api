from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder


class ExtractEntitiesExternalPromptDto(BaseModel):
    user_input: str
    prompt: str
    session_id: str


app = FastAPI()


@app.post("/v3/extract_entities_external_prompt")
async def classify_v3_external_prompt(
    request: ExtractEntitiesExternalPromptDto,
):
    # moderation = get_moderation_intent_entities(
    #     request.user_input, request.session_id
    # )
    # flagged = moderation["flagged"]
    # if flagged:
    #     return moderation

    entities_response = await classify_entities_external_prompt(
        user_input=jsonable_encoder(request.user_input),
        prompt=jsonable_encoder(request.prompt),
    )
    print(entities_response)
    # return {"intent": {"name": intent_response.content, "confidence": 1.0}}
    return entities_response
