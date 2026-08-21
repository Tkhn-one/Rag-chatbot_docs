import os

from dotenv import load_dotenv

load_dotenv()


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Провайдер модели: "openai" или "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Ollama (запускается локально на 11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Параметры разбивки на чанки (в символах)
CHUNK_SIZE = _int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# Сколько фрагментов подаём модели на вопрос
TOP_K = _int("TOP_K", 4)

# Каталоги
DB_DIR = os.getenv("DB_DIR", "db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Температура генерации
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
