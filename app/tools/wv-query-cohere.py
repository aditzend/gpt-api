from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import Cohere
from langchain.chains import ChatVectorDBChain
import weaviate
import json
import requests

import os
from dotenv import load_dotenv

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")

client = weaviate.Client(
    url=weaviate_url,
)

nearText = {"concepts": ["personas"]}

result = (
    client.query.get("Sentence", ["sentence", "category"])
    .with_near_text(nearText)
    .with_limit(2)
    .do()
)

print(json.dumps(result, indent=4))
