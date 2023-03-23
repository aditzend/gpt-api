from langchain import PromptTemplate
from langchain.llms import OpenAI
import json

classify_intent_extract_entities_template = PromptTemplate(
    input_variables=["user_input", "intents", "entities"],
    template="""
    You are an intent and entity classification model.
    Given this list of intents: {intents},
    this list of entities {entities},
    and this user input {user_input}, 
    predict the intent and entities of the user input.

    Give the confidence level of each of your predictions. 
    Choose intents from the list of intents and only from the list of intents.
    Choose entities from the list of entities and only from the list of entities.
    Intent name creation is not allowed.
    Entity name creation is not allowed. If the list of entities 
    is empty, you can't predict any entities. Just predict the intent.
    You have to be really sure about the intent you are predicting, otherwise
    you can respond with the following JSON object :
        "intent"= "Redirigir_BaseDeConocimiento",
        "confidence"= 0.7
 

    Entities are always in the form of a list of dictionaries with this structure :
    "text"= extracted text from user input,"value"= pay attention to the description for indications on how to extract the value of the entity, "name"= name of the entity, "confidence"= confidence level of the prediction

    Intent is always in the form of a dictionary with this structure=
    "name"= name of the intent, "confidence"= confidence level of the prediction

    Only respond with a valid JSON object and nothing else.
    Your response must have this structure:
        "intent"
        "entities"
    """,
)


async def classify_intent_extract_entities(user_input, intents, entities):

    llm = OpenAI(temperature=0.9)
    prompt = classify_intent_extract_entities_template.format(
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
