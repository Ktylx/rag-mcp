"""
Узлы LangGraph графа для RAG-пайплайна.
"""

import logging
from typing import List

from langchain_core.documents import Document

from src import config
from src.indexer.chroma_manager import get_chroma_manager
from src.rag.state import RAGState
from src.rag.prompts import (
    QUERY_REWRITE_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    GENERATION_PROMPT_TEMPLATE,
    HALLUCINATION_CHECK_PROMPT,
    BROADEN_QUERY_PROMPT,
    format_documents_for_prompt,
)
from src.utils.ollama_client import get_llm

logger = logging.getLogger(__name__)


def rewrite_query_node(state: RAGState) -> RAGState:
    """Переформулировать запрос для более эффективного поиска.
    
    NOTE: Rewriting temporarily disabled - using original question directly.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    question = state["question"]
    
    logger.info(f"[rewrite_query_node] Using original question (rewriting disabled): {question}")
    
    # Temporarily disabled - use original question
    rewritten = question
    
    return {
        **state,
        "rewritten_query": rewritten,
    }


def retrieve_node(state: RAGState) -> RAGState:
    """Извлечь документы из векторного хранилища.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    query = state["rewritten_query"]
    chroma = get_chroma_manager()
    
    logger.info(f"[retrieve_node] Searching with query: '{query}' (k={config.DEFAULT_TOP_K})")
    
    # Поиск документов
    documents = chroma.similarity_search(
        query=query,
        k=config.DEFAULT_TOP_K,
    )
    
    # Извлекаем источники
    sources = list(set(
        doc.metadata.get("source", "unknown")
        for doc in documents
    ))
    
    logger.info(f"[retrieve_node] Retrieved {len(documents)} documents: {sources}")
    
    # Логируем превью каждого документа
    for i, doc in enumerate(documents):
        preview = doc.page_content[:150].replace("\n", " ")
        logger.info(f"[retrieve_node] Doc {i+1} preview: {preview}...")
    
    return {
        **state,
        "documents": documents,
        "sources": sources,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    """Оценить релевантность документов к запросу.
    
    Использует LLM для оценки релевантности каждого документа.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm(temperature=0.0)
    
    graded_docs = []
    irrelevant_docs = []
    
    logger.info(f"[grade_documents_node] Grading {len(documents)} documents...")
    
    for i, doc in enumerate(documents):
        # Формируем промпт для оценки
        doc_text = doc.page_content[:1000]  # Ограничиваем длину документа
        
        prompt = GRADE_DOCUMENTS_PROMPT.format(
            question=question,
            document=doc_text,
        )
        
        # Получаем оценку от LLM
        response = llm.invoke(prompt)
        grade = response.content.strip().lower()
        
        is_relevant = "yes" in grade
        
        source = doc.metadata.get("source", "unknown")
        
        if is_relevant:
            graded_docs.append(doc)
            logger.info(f"[grade_documents_node] Doc {i+1} [{source}]: RELEVANT")
        else:
            irrelevant_docs.append(doc)
            logger.info(f"[grade_documents_node] Doc {i+1} [{source}]: NOT RELEVANT")
    
    # Если нет релевантных документов, используем fail-open
    if not graded_docs:
        logger.warning(f"[grade_documents_node] No relevant docs found - using fail-open")
        graded_docs = documents
    
    logger.info(f"[grade_documents_node] Final: {len(graded_docs)} relevant documents (filtered from {len(documents)})")
    
    return {
        **state,
        "graded_documents": graded_docs,
    }


def generate_node(state: RAGState) -> RAGState:
    """Сгенерировать ответ на основе документов.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    question = state["question"]
    documents = state["graded_documents"]
    sources = state["sources"]
    retry_count = state["retry_count"]
    
    if not documents:
        return {
            **state,
            "generation": "Извините, я не нашёл релевантных документов для ответа на ваш вопрос.",
            "is_grounded": False,
        }
    
    llm = get_llm()
    
    # Формируем промпт
    docs_text = format_documents_for_prompt(documents)
    
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        question=question,
        documents=docs_text,
        sources=", ".join(sources),
    )
    
    # Генерируем ответ
    response = llm.invoke(prompt)
    generation = response.content.strip()
    
    logger.info(f"Generated response: {generation[:100]}...")
    logger.info(f"[generate_node] Before: retry_count={retry_count}, After: retry_count={retry_count + 1}")
    
    return {
        **state,
        "generation": generation,
        "retry_count": retry_count + 1,
    }


def hallucination_check_node(state: RAGState) -> RAGState:
    """Проверить ответ на галлюцинации.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    question = state["question"]
    generation = state["generation"]
    documents = state["graded_documents"]
    llm = get_llm(temperature=0.0)
    
    # Если нет документов, считаем ответ неоснованным
    if not documents:
        return {
            **state,
            "is_grounded": False,
        }
    
    # Формируем промпт
    docs_text = format_documents_for_prompt(documents)
    
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        question=question,
        generation=generation,
        documents=docs_text,
    )
    
    # Проверяем на галлюцинации
    response = llm.invoke(prompt)
    check = response.content.strip().lower()
    
    is_grounded = "yes" in check
    
    logger.info(f"Hallucination check: {'passed' if is_grounded else 'failed'}")
    
    return {
        **state,
        "is_grounded": is_grounded,
    }


def broaden_query_node(state: RAGState) -> RAGState:
    """Расширить запрос для поиска большего количества документов.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Обновлённое состояние.
    """
    question = state["question"]
    doc_count = len(state["documents"])
    broaden_count = state["broaden_count"]
    
    # Проверяем лимит циклов
    if broaden_count >= config.MAX_BROADEN_LOOPS:
        logger.info("Max broaden loops reached")
        return state
    
    llm = get_llm()
    
    # Формируем промпт
    prompt = BROADEN_QUERY_PROMPT.format(
        question=question,
        doc_count=doc_count,
    )
    
    # Получаем расширенный запрос
    response = llm.invoke(prompt)
    broadened = response.content.strip()
    
    logger.info(f"Broadened query: {broadened}")
    
    return {
        **state,
        "rewritten_query": broadened,
        "broaden_count": broaden_count + 1,
    }


def should_continue_grade(state: RAGState) -> str:
    """Определить продолжение после оценки документов.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Следующий узел.
    """
    graded_docs = state["graded_documents"]
    
    if len(graded_docs) >= config.MIN_RELEVANT_DOCS:
        return "generate"
    else:
        return "broaden_query"


def should_regenerate(state: RAGState) -> str:
    """Определить, нужно ли регенерировать ответ.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Следующий узел.
    """
    is_grounded = state["is_grounded"]
    retry_count = state["retry_count"]
    broaden_count = state["broaden_count"]
    
    # Детальное логирование для отладки
    logger.info(f"[should_regenerate] retry_count={retry_count}, broaden_count={broaden_count}, is_grounded={is_grounded}")
    
    # Защита от бесконечного цикла - проверяем общий лимит итераций
    total_iterations = retry_count + broaden_count
    if total_iterations >= config.MAX_TOTAL_ITERATIONS:
        logger.warning(f"Max total iterations reached: {total_iterations}")
        return "end"
    
    if is_grounded:
        logger.info("[should_regenerate] -> end (grounded)")
        return "end"
    elif retry_count <= config.MAX_REGENERATE_RETRIES:
        logger.info(f"[should_regenerate] -> generate (retry_count={retry_count} <= {config.MAX_REGENERATE_RETRIES})")
        return "generate"
    else:
        logger.info("[should_regenerate] -> end (max retries)")
        return "end"


def should_broaden(state: RAGState) -> str:
    """Определить, нужно ли расширять запрос.
    
    Args:
        state: Текущее состояние.
        
    Returns:
        Следующий узел.
    """
    broaden_count = state["broaden_count"]
    
    if broaden_count < config.MAX_BROADEN_LOOPS:
        return "retrieve"
    else:
        return "generate"