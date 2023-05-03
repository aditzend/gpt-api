from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import weaviate
import json
import requests

import os
from dotenv import load_dotenv

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

client = weaviate.Client(
    url=weaviate_url, additional_headers={"X-OpenAI-API-KEY": openai_api_key}
)

nearText = {"concepts": ["biology"]}

result = (
    client.query.get("Question", ["question", "answer", "category"])
    .with_near_text(nearText)
    .with_limit(2)
    .do()
)

print(json.dumps(result, indent=4))
