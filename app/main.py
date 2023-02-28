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


PRE_PROMPT = """Having these prompts and their coresponding completions into account:
    

      en movistar me dan wifi 300 megas por $3000, todavia no cerre

      


       {
        "intent": "InformarCompetencia_Baja",
        "entities": [
          {"entity": "ORG","text": "movistar","value":"MOVISTAR"},
          {"entity": "SERVICE","text": "wifi 300 megas","value":"INTERNET_300_MB"},
          {"entity": "PRICE","text": "$3000","value":"3000"},
          {"entity": "RECOVERY_SCORING","text": "","value": 7}
        ]
      }
      

      me cambie a personal me dan 500 mb con un descuento de 30%
      



       {
        "intent": "InformarCompetenciaYDescuento_Baja",
        "entities": [
            {"text":"personal","value":"PERSONAL_FLOW", "entity": "ORG"},
            {"text":"500 mb","value":"INTERNET_500_MB", "entity": "SERVICE"},
            {"text":"30%","value":"30", "entity": "DISCOUNT_PERCENTAGE"},
            {"text":"","value": 5, "entity": "RECOVERY_SCORING"}
          ]
      }

      
      Me ofrecen todo lo mismo en Cablevisión x menos de la mitad de lo que abono acá




       {
        "intent": "InformarCompetenciaYDescuento_Baja",
        "entities": [
            {"text":"Cablevisión Flow","value":"PERSONAL_FLOW", "entity": "ORG"},
            {"text":"todo lo mismo","value":"SAME", "entity": "SERVICE"},
            {"text":"x menos de la mitad","value":"50", "entity": "DISCOUNT_PERCENTAGE"},
            {"text":"","value": 2, "entity": "RECOVERY_SCORING"}
          ]
      }

    What would be the same completion for NEW_PROMPT?

    Only respond with a valid JSON object and nothing else.

    NEW_PROMPT = 'prompt_placeholder'
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
        entities = []
        if "entities" in p:
            entities = p["entities"]
            if entities:
                for entity in entities:
                    entity["start"] = request.text.find(entity["text"])
                    entity["end"] = entity["start"] + len(entity["text"])
                    entity["confidence"] = 1.0
        response = {
            "text": request.text,
            "intent": {
                "id": -999,
                "name": p["intent"],
                "confidence": 1.0,
            },
            "entities": entities,
            "intent_ranking": [],
        }
    return response
