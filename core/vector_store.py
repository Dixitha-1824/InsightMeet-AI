import os
import shutil
import uuid

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:

    print("Building Vector Store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i,
                "source": "meeting_transcript"
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    collection_name = f"meeting_{uuid.uuid4()}"
    vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=collection_name,
    )

    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": 20,
    },
)