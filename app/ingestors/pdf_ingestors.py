import logging
import os
from ingestors.helpers import check_index_existance
import requests
import uuid
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader

logger = logging.getLogger(__name__)


def download_pdf_file_and_ingest(resource):
    logger.info(f"[[download_pdf_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./resources/{index}-{str(uuid.uuid1())}.pdf"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredPDFLoader(local_path)
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
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")
