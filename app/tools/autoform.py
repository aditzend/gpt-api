from langchain.prompts import load_prompt
from langchain import PromptTemplate
from langchain.llms import OpenAI
import json
import os


async def fill_form(user_input, slots_to_fill, slots_filled, personality):
    llm = OpenAI(temperature=0.9)
    autoform = os.path.join(os.path.dirname(__file__), "prompts/autoform.yaml")
    prompt = load_prompt(autoform)
    hydrated = prompt.format(
        user_input=user_input,
        slots_to_fill=slots_to_fill,
        slots_filled=slots_filled,
        personality=personality,
    )
    response = llm(hydrated)
    clean = "{" + response.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response
