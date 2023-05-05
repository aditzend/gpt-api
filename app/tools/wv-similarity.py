from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader
import weaviate

import os
from dotenv import load_dotenv

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
local_file = "/Users/alexander/clients/yzn/gpt-api/app/edenor.txt"

loader = TextLoader(local_file)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


client = weaviate.Client(
    url=weaviate_url,
)

client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written paragraph",
            # "vectorizer": "text2vec-transformers",
            # "moduleConfig": {
            #     "text2vec-openai": {"model": "ada", "type": "text"}
            # },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    # "moduleConfig": {
                    #     "text2vec-openai": {
                    #         "skip": False,
                    #         "vectorizePropertyName": False,
                    #     }
                    # },
                    "name": "content",
                },
            ],
        },
    ]
}

client.schema.create(schema)

vectorstore = Weaviate(client, "Paragraph", "content")

query = "Subsidios"
docs = vectorstore.similarity_search(query)

print(docs)
