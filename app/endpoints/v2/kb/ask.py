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
