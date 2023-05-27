import logging
from ingestors.helpers import check_index_existance
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    YoutubeLoader,
)


logger = logging.getLogger(__name__)


def youtube_ingest(resource):
    logger.info(f"[[download_youtube_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    loader = YoutubeLoader.from_youtube_url(
        url, language="es", add_video_info=True
    )
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
    logger.debug(f"Index {index} saved locally! {index_db.docstore._dict}")
