import uuid

import chromadb
from langchain_chroma import Chroma

COLLECTION = "docs"


def make_store(embeddings, db_dir):
    """Создаёт или открывает векторную базу в папке db_dir."""
    client = chromadb.PersistentClient(path=db_dir)
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )


def add_chunks(vectorstore, chunks):
    """Добавляет чанки в базу. Возвращает количество добавленных."""
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "loc": c["loc"]} for c in chunks]
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(ids)


def search(vectorstore, question, k):
    """Ищет top-k фрагментов по вопросу. Возвращает (docs, scores)."""
    return vectorstore.similarity_search_with_relevance_scores(question, k=k)


def delete_source(vectorstore, source):
    """Удаляет все чанки, привязанные к конкретному файлу."""
    vectorstore.delete(where={"source": source})


def collection_size(vectorstore):
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def known_sources(vectorstore):
    try:
        data = vectorstore._collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"]})
    except Exception:
        return []
