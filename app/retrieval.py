from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(r"D:\code\nlp\ta-sisdas")
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "bge-m3"


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

def retrieve(vectorstore, query: str, metadata_filter: dict, k: int = 10):
    if metadata_filter:
        return vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=metadata_filter
        )

    return vectorstore.similarity_search_with_score(
        query,
        k=k
    )
