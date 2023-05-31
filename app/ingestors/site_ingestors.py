import logging
import os
from ingestors.helpers import check_index_existance
import requests
import uuid
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
import tiktoken

logger = logging.getLogger("uvicorn")

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def ingest_site(resource):
    logger.info(f"[[ingest_site]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)
    main_index_exists: bool = check_index_existance("main")

    # local_path = f"./resources/{index}-{str(uuid.uuid1())}.html"

    # response = requests.get(url)
    # with open(local_path, "wb") as f:
    #     f.write(response.content)
    # logger.info(f"File downloaded successfully to {local_path} !")

    loader = WebBaseLoader(url)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(doc)
    logger.debug(f"Chunk 0 : {chunks[0]}")
    logger.debug(f"Chunk 1 : {chunks[1]}")
    embeddings = OpenAIEmbeddings()
    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(f"indexes/{index}")
    logger.info(f"Index {index} saved ")
