import logging
import os
from ingestors.helpers import check_index_existance
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
)


logger = logging.getLogger(__name__)


def download_docx_file_and_ingest(resource):
    logger.info(f"[[download_docx_file_and_ingest]] Starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./resources/{index}-{os.path.basename(url)}"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(
        "[[ download_docx_file_and_ingest ]] File downloaded successfully to"
        f" {local_path} !"
    )

    loader = UnstructuredWordDocumentLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(f"indexes/{index}")
    os.remove(local_path)
    logger.info(
        f"[[ download_docx_file_and_ingest ]] Index {index} saved locally!"
    )
