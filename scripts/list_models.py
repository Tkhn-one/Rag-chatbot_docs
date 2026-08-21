"""Выводит список моделей, доступных по ключу из .env.

Использует тот же способ подключения, что и приложение (httpx), поэтому
не зависит от curl и обходит проблемы с сертификатами в Windows.

Запуск: python scripts/list_models.py
"""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = (os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

if not api_key:
    print("В .env не задан ключ (LLM_API_KEY или OPENAI_API_KEY).")
    sys.exit(1)

resp = httpx.get(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
resp.raise_for_status()

ids = [m["id"] for m in resp.json()["data"]]
print(f"Базовый URL: {base_url}")
print(f"Доступных моделей: {len(ids)}\n")
for mid in sorted(ids):
    print(" -", mid)
