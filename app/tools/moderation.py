import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_moderation_intent_entities(user_input, session_id):
    moderation = openai.Moderation.create(input=user_input)
    flagged = moderation["results"][0]["flagged"]
    if flagged:
        flagged_response = {
            "flagged": flagged,
            "session_id": session_id,
            "user_input": user_input,
            "intent": {"name": "Bloquear_Moderacion", "confidence": 1.0},
            "entities": [
                {
                    "name": "lenguaje_sexual_detectado",
                    "value": moderation["results"][0]["categories"]["sexual"],
                },
                {
                    "name": "lenguaje_violento_detectado",
                    "value": moderation["results"][0]["categories"][
                        "violence"
                    ],
                },
                {
                    "name": "lenguaje_de_odio_detectado",
                    "value": moderation["results"][0]["categories"]["hate"],
                },
                {
                    "name": "lenguaje_autoflagelante_detectado",
                    "value": moderation["results"][0]["categories"][
                        "self-harm"
                    ],
                },
                {
                    "name": "lenguaje_de_sexual_menores_detectado",
                    "value": moderation["results"][0]["categories"][
                        "sexual/minors"
                    ],
                },
                {
                    "name": "lenguaje_de_odio_amenaza_detectado",
                    "value": moderation["results"][0]["categories"][
                        "hate/threatening"
                    ],
                },
                {
                    "name": "lenguaje_de_violencia_grafica_detectado",
                    "value": moderation["results"][0]["categories"][
                        "violence/graphic"
                    ],
                },
            ],
        }
        return flagged_response
    else:
        return {
            "flagged": False,
        }
