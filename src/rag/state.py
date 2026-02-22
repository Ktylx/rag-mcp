"""
Типы данных для состояния RAG-графа.
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    """Состояние RAG-графа.
    
    Атрибуты:
        question: Исходный вопрос пользователя.
        rewritten_query: Переформулированный запрос для поиска.
        documents: Найденные документы из векторного хранилища.
        graded_documents: Отфильтрованные релевантные документы.
        generation: Сгенерированный ответ.
        sources: Список источников (путей к файлам).
        retry_count: Счётчик попыток регенерации.
        broaden_count: Счётчик циклов расширения запроса.
        is_grounded: Флаг, указывающий, что ответ основан на документах.
        error: Сообщение об ошибке (если есть).
    """
    question: str
    rewritten_query: str
    documents: List[Document]
    graded_documents: List[Document]
    generation: str
    sources: List[str]
    retry_count: int
    broaden_count: int
    is_grounded: bool
    error: Optional[str]


def create_initial_state(question: str) -> RAGState:
    """Создать начальное состояние для RAG-пайплайна.
    
    Args:
        question: Вопрос пользователя.
        
    Returns:
        Начальное состояние.
    """
    return RAGState(
        question=question,
        rewritten_query=question,
        documents=[],
        graded_documents=[],
        generation="",
        sources=[],
        retry_count=0,
        broaden_count=0,
        is_grounded=False,
        error=None,
    )