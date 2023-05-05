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

# ===== import data =====
# Load data

# url = "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
url = "https://assets.saia.ar/yoizen/edenor.json"
resp = requests.get(url)
data = json.loads(resp.text)

# Configure a batch process
with client.batch as batch:
    batch.batch_size = 100
    # Batch import all Questions
    for i, d in enumerate(data):
        print(f"importing sentence: {i+1}")

        properties = {
            "sentence": d["Sentence"],
            "category": d["Category"],
        }

        client.batch.add_data_object(properties, "Sentence")
