from pathlib import Path

import pypdf
from docx import Document

SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}
MAX_FILE_BYTES = 20 * 1024 * 1024


class LoadError(Exception):
    """Ошибка чтения или проверки документа."""


def read_document(path):
    """Читает документ и возвращает список страниц/абзацев.

    Каждый элемент — словарь с ключами: text, source, loc.
    loc — человекочитаемое место в документе (номер страницы или абзаца).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        raise LoadError(f"Неподдерживаемый формат «{ext}». Разрешены: PDF, DOCX, TXT")

    size = path.stat().st_size
    if size == 0:
        raise LoadError("Файл пустой — текста для индексации нет")
    if size > MAX_FILE_BYTES:
        raise LoadError(f"Файл больше {MAX_FILE_BYTES // 1024 // 1024} МБ")

    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    return _read_txt(path)


def _read_pdf(path):
    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:
        raise LoadError(f"Не удалось открыть PDF: {exc}") from exc

    if reader.is_encrypted:
        raise LoadError("PDF защищён паролем — открыть его не получится")

    pages = []
    for num, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())
        if text:
            pages.append({"text": text, "source": path.name, "loc": f"Стр. {num}"})

    if not pages:
        raise LoadError("В PDF не оказалось текстового слоя. Похоже, это сканы без OCR")
    return pages


def _read_docx(path):
    try:
        doc = Document(path)
    except Exception as exc:
        raise LoadError(f"Не удалось открыть DOCX: {exc}") from exc

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise LoadError("В DOCX не оказалось текста (возможно, это только таблицы или картинки)")
    return [
        {"text": text, "source": path.name, "loc": f"Абзац {idx}"}
        for idx, text in enumerate(paragraphs, 1)
    ]


def _read_txt(path):
    raw = path.read_bytes()
    text = _decode_text(raw)
    text = text.strip()
    if not text:
        raise LoadError("Файл TXT пустой или не содержит текста")

    # В TXT нет страниц, поэтому собираем абзацы по пустым строкам
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [text]
    return [
        {"text": block, "source": path.name, "loc": f"Абзац {idx}"}
        for idx, block in enumerate(blocks, 1)
    ]


def _decode_text(raw):
    # cp1251 — кириллица в Windows, часто попадается в TXT от бухгалтерии и юристов
    for enc in ("utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # Если ни одна кодировка не подошла, последняя попытка. Это почти всегда сработает,
    # но может испортить не-латинские символы — лучше, чем упасть с ошибкой.
    return raw.decode("latin-1", errors="replace")
