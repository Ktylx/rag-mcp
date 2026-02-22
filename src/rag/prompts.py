"""
Промпты для RAG-графа.
"""

from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src import config


# Промпт для переформулирования запроса
QUERY_REWRITE_PROMPT = PromptTemplate.from_template(
    """Переформулируй следующий запрос так, чтобы он был более эффективным для поиска в базе знаний.

Текущий запрос: {question}

Переформулированный запрос:"""
)


# Промпт для оценки релевантности документов
GRADE_DOCUMENTS_PROMPT = PromptTemplate.from_template(
    """Определи, релевантен ли каждый документ к запросу пользователя.

Запрос: {question}

Документ: {document}

Ответь ТОЛЬКО одним словом: "yes" если документ релевантен, "no" если нет."""
)


# Промпт для генерации ответа
def create_generation_prompt(documents: List[Document], sources: List[str]) -> str:
    """Создать промпт для генерации ответа.
    
    Args:
        documents: Список документов.
        sources: Список источников.
        
    Returns:
        Сформированный промпт.
    """
    docs_text = "\n\n---\n\n".join(
        f"Документ {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )
    sources_text = ", ".join(sources)
    
    return f"""Ты - ассистент, который отвечает на вопросы на основе предоставленных документов.

Используй ТОЛЬКО информацию из предоставленных документов. Если информации недостаточно, честно скажи об этом.

Запрос: {{question}}

Документы:
{docs_text}

Ответь на запрос на основе документов. После ответа укажи источники в формате:
Источники: {sources_text}"""


GENERATION_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Ты - ассистент, который отвечает на вопросы на основе предоставленных документов.

Используй ТОЛЬКО информацию из предоставленных документов. Если информации недостаточно, честно скажи об этом.

Запрос: {question}

Документы:
{documents}

Ответь на запрос на основе документов. После ответа укажи источники в формате:
Источники: {sources}"""
)


# Промпт для проверки на галлюцинации
HALLUCINATION_CHECK_PROMPT = PromptTemplate.from_template(
    """Проверь, основан ли сгенерированный ответ на предоставленных документах.

Запрос: {question}

Сгенерированный ответ: {generation}

Документы:
{documents}

Ответь ТОЛЬКО одним словом: "yes" если ответ основан на документах, "no" если содержит информацию не из документов."""
)


# Промпт для расширения запроса
BROADEN_QUERY_PROMPT = PromptTemplate.from_template(
    """Расширь и переформулируй запрос, чтобы найти больше релевантных документов.

Текущий запрос: {question}
Найдено документов: {doc_count}

Расширенный запрос:"""
)


# Промпт для саммари
SUMMARIZE_PROMPT = PromptTemplate.from_template(
    """Создай краткое саммари предоставленного документа.

Документ: {document}

Саммари:"""
)


def format_documents_for_prompt(documents: List[Document]) -> str:
    """Отформатировать документы для промпта.
    
    Args:
        documents: Список документов.
        
    Returns:
        Отформатированная строка.
    """
    return "\n\n---\n\n".join(
        f"Документ {i+1} (источник: {doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )