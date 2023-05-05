from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()


# replace
ZILLIZ_CLOUD_URI = (  # example: "https://in01-17f69c292d4a5sa.aws-us-west-2.vectordb.zillizcloud.com:19536"
    "https://in01-3586fcc33673966.aws-us-east-2.vectordb.zillizcloud.com:19541"
)
ZILLIZ_CLOUD_USERNAME = "db_admin"  # example: "username"
ZILLIZ_CLOUD_PASSWORD = "Bvcgfdtre543"  # example: "*********"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader


local_file = "/Users/alexander/clients/yzn/gpt-api/app/edenor.txt"

loader = TextLoader(local_file)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={
        "uri": ZILLIZ_CLOUD_URI,
        "username": ZILLIZ_CLOUD_USERNAME,
        "password": ZILLIZ_CLOUD_PASSWORD,
        "secure": True,
    },
)

# docs = vector_db.similarity_search(query)

# print(docs[0])
