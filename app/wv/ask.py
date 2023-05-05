from langchain.vectorstores.weaviate import Weaviate
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

ask = {"question": "y del subsidio que paso?", "properties": ["sentence"]}

result = (
    client.query.get(
        "Sentence",
        [
            "sentence",
            (
                "_additional {answer {hasAnswer certainty property result"
                " startPosition endPosition} }"
            ),
        ],
    )
    .with_ask(ask)
    .with_limit(1)
    .do()
)

print(result)
