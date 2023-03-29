from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

app = FastAPI()


class SlotToFill(BaseModel):
    name: str
    description: str


class SlotFilled(BaseModel):
    name: str
    value: str


class AutoformDto(BaseModel):
    session_id: str
    user_input: str
    slots_to_fill: list[SlotToFill]
    slots_filled: list[SlotFilled]
    personality: str


@app.post("/v2/autoform")
async def autoformv2(body: AutoformDto):
    form_response = await cleaned_form(
        user_input=jsonable_encoder(body.user_input),
        slots_to_fill=jsonable_encoder(body.slots_to_fill),
        slots_filled=jsonable_encoder(body.slots_filled),
        personality=jsonable_encoder(body.personality),
    )
    # response = {
    #     "session_id": body.session_id,
    #     "user_input": body.user_input,
    #     "sentiment": seh["sentiment"],
    #     "emotion": seh["emotion"],
    #     "hate_speech": seh["hate_speech"],
    #     **form_response,
    # }
    return form_response
