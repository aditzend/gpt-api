from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tools.openaidirect import (
    system_completion_v1_turbo_t0,
    system_user_v1_turbo_t0,
)

import os
from dotenv import load_dotenv

load_dotenv()


classify_intent_extract_entities_template_1 = PromptTemplate(
    input_variables=["user_input", "intents", "entities"],
    template="""
Texto de entrada: "{user_input}"

Prompt: 
Dado el siguiente texto de entrada, identificar la intención del usuario e identificar las entidades relevantes utilizando solamente el array entities. 
Si intención no se encuentra en la lista proporcionada, devolver la intención Redirigir_BaseDeConocimiento.
Solo se pueden utilizar las entities informadas, no se pueden crear o inventar nuevas.
La lista de entidades puede ser vacía si no se encontró nada. No poner entidades con nombre "none"

confidence es un número entre 0 y 1 que indica la confianza del modelo en su predicción.

text se refiere al texto original que se utilizó para identificar la entidad.

Intents:
{intents}

entities:
{entities}

La salida del prompt: 

Solo responder con un formato JSON valido y nada más. El json deberá tener los campos intent=name,confidence
entities=name,value,text,confidence
    """,
)


classify_intent_extract_entities_template_2 = PromptTemplate(
    input_variables=["user_input", "intents", "entities"],
    template="""
Input Text: "{user_input}"

Intents:
{intents}

Entities:
{entities}

Prompt:
Given the input text, determine the user's intent and identify relevant entities using only the entities array. If the intent is not found in the provided list, return the Redirect_KnowledgeBase intent. Only use the given entities; do not create or invent new ones. Think step by step. Ask yourself if every value you extract matches the corresponding description before you extract it.

You have to be very sure about the value you assign to an entity. Use the entity description to help you.


Prompt Output:

Only respond with a valid JSON format and nothing else. The JSON should have the fields intent => name, confidence
entities => name,value,text,confidence
    """,
)

classify_intent_extract_entities_template_3 = PromptTemplate(
    input_variables=["user_input", "intents", "entities"],
    template="""
    Given this list of intents: {intents},
    this list of entities {entities},
    and this user input {user_input}, 
    predict the intent and entities of the user input.

    Give the confidence level of each of your predictions. 

    You have to be really sure about the intent you are predicting, otherwise
    you can respond with the following JSON object :
        "intent"= "Redirigir_BaseDeConocimiento",
        "confidence"= 0.7

    Choose intents from the list of intents and only from the list of intents.
    Choose entities from the list of entities and only from the list of entities.
    Intent name creation is not allowed.
    Entity name creation is not allowed. If the list of entities 
    is empty, you can't predict any entities. Just predict the intent.




    Entities are always in the form of a list of dictionaries with this structure :
    "text"= extracted text from user input,"value"= value extracted, "name"= name of the entity, "confidence"= confidence level of the prediction

    Intent is always in the form of a dictionary with this structure=
    "name"= name of the intent, "confidence"= confidence level of the prediction.

    You have to be really sure about the intent you are predicting, otherwise
    you can respond with the following JSON object :
        "intent"= "Redirigir_BaseDeConocimiento",
        "confidence"= 0.7

    Only respond with a valid JSON object and nothing else.
    Do not explain your reasoning, just predict the intent and entities.
    Your response must have this structure:
        "intent"
        "entities"
    """,
)


async def classify_intent_external_prompt(user_input, prompt):
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_input),
    ]
    chat = ChatOpenAI(verbose=True, temperature=0)
    openai_response = chat(messages)
    response = json.loads(openai_response.content)
    return response


async def classify_entities_external_prompt(user_input, prompt):
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_input),
    ]
    chat = ChatOpenAI(verbose=True, temperature=0)
    openai_response = chat(messages)
    response = json.loads(openai_response.content)
    return response


async def classify_intent_extract_entities(user_input, intents, entities):
    chat = ChatOpenAI(verbose=True, temperature=0)
    prompt = classify_intent_extract_entities_template_2.format(
        user_input=user_input,
        intents=intents,
        entities=entities,
    )
    messages = [
        SystemMessage(
            content="You are an intent and entity classification model."
        ),
        HumanMessage(content=prompt),
    ]

    # print(openai_response.content)
    response = json.loads(openai_response.content)
    # clean = "{" + openai_response.split("{", 1)[1]
    # clean = clean.replace("'", '"')
    # print(clean)
    # response = json.loads(clean)
    return response


async def classify_intent_extract_entities_creative(
    user_input, intents, entities
):
    llm = OpenAI(temperature=0.9)
    prompt = classify_intent_extract_entities_creative_template.format(
        user_input=user_input,
        intents=intents,
        entities=entities,
    )
    response = llm(prompt)
    clean = "{" + response.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response


async def classify_intent_extract_entities_english(
    user_input, intents, entities
):
    llm = OpenAI(temperature=0.9)
    prompt = classify_intent_extract_entities_english_template.format(
        user_input=user_input,
        intents=intents,
        entities=entities,
    )
    response = llm(prompt)
    print(response)
    clean = "{" + response.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response


async def bulk_intent_classifier(intents, user_input):
    system = (
        "You are an intent classification model. INTENTS:"
        f" {intents}\n###\nChoose the most appopiate intent for the user"
        " message. Just respond with the intent name."
    )
    user = user_input
    response = await system_user_v1_turbo_t0(system=system, user=user)
    return response
