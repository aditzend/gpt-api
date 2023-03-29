class AutoformExternalPromptDto(BaseModel):
    user_input: str
    prompt: str
    session_id: str


@app.post("/v2/extract_intent_and_entities")
async def classify_v2(request: ClassifyIntentExtractEntitiesV2Dto):
    moderation = get_moderation_intent_entities(
        request.user_input, request.session_id
    )
    flagged = moderation["flagged"]
    if flagged:
        return moderation

    classify_response: IntentEntitiesDto = (
        await classify_intent_extract_entities(
            user_input=jsonable_encoder(request.user_input),
            intents=jsonable_encoder(request.intents),
            entities=jsonable_encoder(request.entities),
        )
    )
    seh_response = await sentiment_emotion_hate(request.user_input)
    print(classify_response)
    response = ""
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
