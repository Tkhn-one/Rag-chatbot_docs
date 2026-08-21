import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .store import search

SYSTEM_PROMPT = (
    "Ты ассистент, который отвечает на вопросы только по загруженным документам. "
    "Используй фрагменты из «Контекста». Если в контексте нет ответа — честно скажи, "
    "что по документам этого найти не удалось, и не выдумывай. "
    "После ответа указывай источники в квадратных скобках: [1], [2] и так далее — "
    "номер соответствует фрагменту из контекста. Не упоминай источники, которых не использовал."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
    ]
)

_CITE_RE = re.compile(r"\[(\d+)\]")


def _raise_helpful(exc):
    """Превращает типичные ошибки API в понятные сообщения."""
    text = str(exc)
    if "model_not_found" in text or "does not exist" in text:
        raise RuntimeError(
            "Имя модели не найдено. Открой файл .env и проверь LLM_MODEL — "
            "такой модели нет у выбранного сервиса (Groq/OpenRouter и т.п.). "
            "Список доступных моделей смотри на сайте сервиса или в его консоли."
        ) from exc
    if "429" in text or "rate limit" in text.lower() or "quota" in text.lower():
        raise RuntimeError(
            "Превышен лимит запросов к API (429). Подожди немного или проверь "
            "бесплатные лимиты выбранного сервиса."
        ) from exc
    if "authentication" in text.lower() or "invalid api key" in text.lower() or "401" in text:
        raise RuntimeError(
            "Неверный API-ключ. Проверь LLM_API_KEY / EMBEDDING_API_KEY в файле .env."
        ) from exc
    raise exc


def answer(llm, vectorstore, history, question, k):
    """Ищет релевантные фрагменты и генерирует ответ с источниками.

    Возвращает (текст_ответа, список_источников), где источник — словарь
    {n, source, loc}.
    """
    hits = search(vectorstore, question, k)

    context = []
    labels = []
    for idx, (doc, _score) in enumerate(hits, 1):
        context.append(f"[{idx}] {doc.page_content}")
        labels.append({"n": idx, "source": doc.metadata.get("source"), "loc": doc.metadata.get("loc")})

    messages = PROMPT.format_messages(
        history=history[-8:],  # держим историю короткой, чтобы не раздувать запрос
        context="\n\n".join(context) if context else "Контекст пуст.",
        question=question,
    )

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        _raise_helpful(exc)
    answer_text = response.content

    sources = []
    seen = set()
    for n in _CITE_RE.findall(answer_text):
        n = int(n)
        if n in seen or not (1 <= n <= len(labels)):
            continue
        seen.add(n)
        sources.append(labels[n - 1])
    return answer_text, sources
