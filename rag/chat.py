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

    response = llm.invoke(messages)
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
