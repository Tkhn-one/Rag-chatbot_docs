def split_pages(pages, size, overlap):
    """Разбивает страницы/абзацы на чанки, сохраняя метаданные источника."""
    chunks = []
    for item in pages:
        parts = _slice(item["text"], size, overlap)
        for idx, part in enumerate(parts, 1):
            loc = item["loc"]
            if len(parts) > 1:
                loc = f"{item['loc']}, часть {idx}"
            chunks.append({"text": part, "source": item["source"], "loc": loc})
    return chunks


def _slice(text, size, overlap):
    """Режет текст на куски размером ~size символов с перекрытием overlap."""
    if len(text) <= size:
        return [text]

    parts = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        # Не рвать слово на границе чанка
        if end < length:
            end = _word_boundary(text, end)
        parts.append(text[start:end].strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return parts


def _word_boundary(text, pos):
    # Ищем ближайший пробел вправо от pos, чтобы не резать посреди слова
    nxt = text.find(" ", pos)
    if nxt == -1 or nxt - pos > 40:
        return pos
    return nxt
