"""
Текстовый сплиттер для разбиения документов на чанки.
Поддерживает разные типы контента: текст и код.
"""

from typing import List, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_core.documents import Document

from src import config


# Языки для сплиттера кода
CODE_LANGUAGES = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
}


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Получить сплиттер для текстовых файлов.
    
    Args:
        chunk_size: Размер чанка (по умолчанию из конфига).
        chunk_overlap: Перекрытие чанков (по умолчанию из конфига).
        
    Returns:
        RecursiveCharacterTextSplitter для текста.
    """
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n",   # Абзацы
            "\n",     # Строки
            ". ",     # Предложения
            ", ",     # Фразы
            " ",      # Слова
        ],
        is_separator_regex=False,
    )


def get_code_splitter(
    language: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Получить сплиттер для кода.
    
    Args:
        language: Язык программирования.
        chunk_size: Размер чанка (по умолчанию из конфига).
        chunk_overlap: Перекрытие чанков (по умолчанию из конфига).
        
    Returns:
        RecursiveCharacterTextSplitter для кода.
    """
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    # Получаем enum языка
    lang_enum = CODE_LANGUAGES.get(language.lower())
    if lang_enum is None:
        # Если язык не известен, используем текстовый сплиттер
        return get_text_splitter(size, overlap)
    
    return RecursiveCharacterTextSplitter.from_language(
        language=lang_enum,
        chunk_size=size,
        chunk_overlap=overlap,
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Разбить документы на чанки.
    
    Args:
        documents: Список документов.
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие чанков.
        
    Returns:
        Список чанков (документов).
    """
    if not documents:
        return []
    
    # Группируем документы по типу (код vs текст)
    code_docs = []
    text_docs = []
    
    for doc in documents:
        ext = doc.metadata.get("file_extension", "")
        if ext in CODE_LANGUAGES:
            code_docs.append(doc)
        else:
            text_docs.append(doc)
    
    chunks = []
    
    # Обрабатываем текстовые документы
    if text_docs:
        text_splitter = get_text_splitter(chunk_size, chunk_overlap)
        text_chunks = text_splitter.split_documents(text_docs)
        chunks.extend(text_chunks)
    
    # Обрабатываем кодовые документы
    if code_docs:
        # Группируем по языку
        by_language: dict[str, List[Document]] = {}
        for doc in code_docs:
            ext = doc.metadata.get("file_extension", "")
            if ext not in by_language:
                by_language[ext] = []
            by_language[ext].append(doc)
        
        for ext, docs in by_language.items():
            code_splitter = get_code_splitter(ext, chunk_size, chunk_overlap)
            code_chunks = code_splitter.split_documents(docs)
            chunks.extend(code_chunks)
    
    # Добавляем индекс чанка в метаданные
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks