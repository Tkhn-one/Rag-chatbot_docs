import httpx


class OpenAICompatEmbeddings:
    """Клиент для любого OpenAI-совместимого эмбеддинг-API (OpenAI, Jina и т.п.).

    Отправляет текст как есть (строки), без токенизации, поэтому работает и с
    сервисами, которые не понимают массивы токенов (например, Jina).
    """

    def __init__(self, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

    def _embed(self, texts):
        payload = {"model": self.model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]
