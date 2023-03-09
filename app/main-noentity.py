from fastapi import FastAPI
from pydantic import BaseModel
import json
import re
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


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


PRE_PROMPT = """
    You are an intent classification model.
    Given the following list of intents INTENT_LIST and a user input USER_INPUT, 
    predict the intent of the user input.

    Give the confidence level of each of your predictions. 
    Choose intents from the list of intents and only from the list of intents.
    Intent name creation is not allowed.
    You have to be really sure about the intent you are predicting, otherwise
    you can respond with the following JSON object:
    {
        "intent": "kb",
        "confidence": 0.7
    }

    INTENT_LIST
    [
        "InformariMisPuntos_InfraccionesTransito",
        "InformarCostoLicencia_LicenciasDeConducir",
        "InformarInfracciones_InfraccionesTransito",
        "InformarActualizar_LicenciasDeConducir",
        "InformarCursovial_LicenciasDeConducir",
        "InformarEstadoLicencia_LicenciasDeConducir",
        "InformarExtranjero_LicenciasDeConducir",
        "InformarLegalizar_LicenciasDeConducir",
        "InformarLibreDeuda_InfraccionesTransito",
        "InformarMudanza_LicenciasDeConducir",
        "InformarRobo_LicenciasDeConducir",
        "InformarVacunas_Vacunas",
        "OtorgarTramiteLicencia_LicenciasDeConducir",
        "Saludo",
    ]

    Only respond with a valid JSON object and nothing else.

    USER_INPUT = 'prompt_placeholder'

    Think step by step and be very careful with your predictions.
    Go!
    """


class RasaModelParseRequest(BaseModel):
    text: str
    message_id: str
    lang: str


@app.post("/model/parse")
async def parse_text(request: RasaModelParseRequest):

    pre_prompt = PRE_PROMPT

    prompt = pre_prompt.replace("prompt_placeholder", request.text)

    openai_response = model_prediction(text=prompt, model="text-davinci-003")

    response = openai_response
    if openai_response:
        p = json.loads(openai_response)
        print(type(p))
        print(p)
        response = {
            "text": request.text,
            "intent": {
                "id": -999,
                "name": p["intent"],
                "confidence": p["confidence"],
            },
            "entities": [],
            "intent_ranking": [],
        }
    return response
