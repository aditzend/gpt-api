from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader
import weaviate

import os
from dotenv import load_dotenv

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader("../docs/osde.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


client = weaviate.Client(
    url=weaviate_url, additional_headers={"X-OpenAI-Api-Key": openai_api_key}
)

client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written paragraph",
            "vectorizer": "text2vec-transformers",
            # "moduleConfig": {
            #     "text2vec-openai": {"model": "ada", "type": "text"}
            # },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}

client.schema.create(schema)

vectorstore = Weaviate(client, "Paragraph", "content")

query = "reintegros?"
docs = vectorstore.similarity_search(query)

print(docs)
