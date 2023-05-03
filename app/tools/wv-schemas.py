from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import weaviate

import os
from dotenv import load_dotenv

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

client = weaviate.Client(
    url=weaviate_url, additional_headers={"X-OpenAI-API-KEY": openai_api_key}
)

print(client.schema.get())

class_obj = {
    "class": "Question",
    "vectorizer": (  # Or "text2vec-cohere" or "text2vec-huggingface"
        "text2vec-openai"
    ),
}

client.schema.create_class(class_obj)