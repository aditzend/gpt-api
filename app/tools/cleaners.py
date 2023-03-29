from langchain import PromptTemplate
from langchain.llms import OpenAI
import json
from pydantic import BaseModel


json_extractor_template = PromptTemplate(
    input_variables=["dirty_json"],
    template="""
    Extract the response as a valid JSON object

    {dirty_json}
    """,
)


async def extract_json(dirty_json):
    # llm = OpenAI(temperature=0.0)
    # prompt = json_extractor_template.format(text=dirty_json)
    # response = llm(prompt)
    # response = json.loads(response)
    return "ok"
