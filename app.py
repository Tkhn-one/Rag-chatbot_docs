import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_DIR,
    EMBEDDING_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    TOP_K,
    UPLOAD_DIR,
)
from rag import chat, chunking, loaders, store
from rag.embeddings import get_embeddings, get_llm, provider_name

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

st.set_page_config(page_title="RAG-чатбот по документам", page_icon="📚", layout="centered")


@st.cache_resource
def _store():
    return store.make_store(get_embeddings(), DB_DIR)


@st.cache_resource
def _llm():
    return get_llm()


def _index_file(filename, raw):
    """Сохраняет файл, читает, режет на чанки и кладёт в базу."""
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as fh:
        fh.write(raw)

    pages = loaders.read_document(path)
    chunks = chunking.split_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    vectorstore = _store()
    store.delete_source(vectorstore, filename)  # переиндексация вместо дублей
    added = store.add_chunks(vectorstore, chunks)
    return len(pages), added


def _render_sources(msg):
    for s in msg.get("sources", []):
        st.caption(f"📄 {s['source']} — {s['loc']}")


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _history():
    msgs = []
    for m in st.session_state.messages:
        msgs.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    return msgs


_init_state()
vectorstore = _store()
llm = _llm()

# ---------- Боковая панель ----------
with st.sidebar:
    st.title("📚 Документы")
    if LLM_PROVIDER == "ollama":
        model_label = f"Ollama ({OLLAMA_MODEL})"
    elif LLM_BASE_URL:
        model_label = f"{LLM_BASE_URL} · {LLM_MODEL}"
    else:
        model_label = f"OpenAI ({LLM_MODEL})"
    st.caption(f"Провайдер: {provider_name()} · {model_label}")
    if LLM_PROVIDER == "openai" and not (LLM_API_KEY or EMBEDDING_API_KEY):
        st.error("Ключи API не заданы — впиши их в файл .env")

    uploaded = st.file_uploader(
        "Загрузи файлы (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Индексировать загруженные", use_container_width=True, disabled=not uploaded):
        indexed = 0
        with st.spinner("Читаю и раскладываю по базе..."):
            for file in uploaded:
                try:
                    _index_file(file.name, file.getvalue())
                    indexed += 1
                except loaders.LoadError as exc:
                    st.error(f"{file.name}: {exc}")
                except Exception as exc:
                    st.error(f"{file.name}: неожиданная ошибка — {exc}")
        st.success(f"Готово: {indexed} из {len(uploaded)} файлов добавлены.")

    st.divider()
    st.subheader("Индексированные файлы")
    files = store.known_sources(vectorstore)
    if files:
        for name in files:
            col1, col2 = st.columns([4, 1])
            col1.write(name)
            if col2.button("✕", key=f"del_{name}", help=f"Удалить {name}"):
                store.delete_source(vectorstore, name)
                st.rerun()
    else:
        st.caption("Пока пусто — загрузи файл выше.")

    st.divider()
    if st.button("🗑️ Очистить всю базу", use_container_width=True):
        for name in store.known_sources(vectorstore):
            store.delete_source(vectorstore, name)
        st.rerun()

    if st.button("♻️ Новый диалог", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Чат ----------
st.title("Спроси по своим документам")
st.caption("Пиши вопрос — ответ придёт с указанием страницы или абзаца.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg)

if prompt := st.chat_input("Твой вопрос по документам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в документах и готовлю ответ..."):
            try:
                answer_text, sources = chat.answer(llm, vectorstore, _history(), prompt, TOP_K)
            except Exception as exc:
                answer_text = f"Не получилось ответить. Проверь, что модель запущена: {exc}"
                sources = []
        st.markdown(answer_text)
        if sources:
            st.markdown("---")
            st.markdown("**Источники:**")
            for s in sources:
                st.markdown(f"- **[{s['n']}]** {s['source']} — {s['loc']}")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
