from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TEMPERATURE,
)


def get_llm():
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        # base_url None означает стандартный api.openai.com
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY or OPENAI_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=TEMPERATURE,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def get_embeddings():
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        # Эмбеддинги могут жить на другом сервисе, чем генерация
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY or OPENAI_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def provider_name():
    return LLM_PROVIDER
