from fastapi.encoders import jsonable_encoder


class Intent(BaseModel):
    name: str
    description: str


class Entity(BaseModel):
    name: str
    description: str


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
