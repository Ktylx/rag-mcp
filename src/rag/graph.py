"""
LangGraph граф для RAG пайплайна (Corrective RAG).
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from src.rag.state import RAGState, create_initial_state
from src.rag.nodes import (
    rewrite_query_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    hallucination_check_node,
    broaden_query_node,
    should_continue_grade,
    should_regenerate,
    should_broaden,
)

logger = logging.getLogger(__name__)

def create_rag_graph() -> StateGraph:
    """Создать LangGraph граф для RAG пайплайна (Corrective RAG).
    
    Flow:
    1. rewrite_query - переформулировать запрос
    2. retrieve - извлечь документы
    3. grade_documents - оценить релевантность
    4a. enough relevant → generate
    4b. too few → broaden_query → retrieve (loop)
    5. generate - сгенерировать ответ
    6. hallucination_check - проверить на галлюцинации
    7a. grounded → end
    7b. not grounded → regenerate (max 1 retry)
    
    Returns:
        Скомпилированный граф.
    """
    # Создаём новый граф при каждом вызове
    workflow = StateGraph(RAGState)
    
    # Добавляем все узлы (полная схема Corrective RAG)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("broaden_query", broaden_query_node)
    
    # Устанавливаем начальный узел
    workflow.set_entry_point("rewrite_query")
    
    # rewrite → retrieve
    workflow.add_edge("rewrite_query", "retrieve")
    
    # retrieve → grade_documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # grade_documents → conditional (generate или broaden_query)
    # Если enough relevant → generate, иначе → broaden_query
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue_grade,
        {
            "generate": "generate",
            "broaden_query": "broaden_query",
        }
    )
    
    # broaden_query → conditional (retrieve или generate)
    # Если не превышен лимит циклов → retrieve, иначе → generate
    workflow.add_conditional_edges(
        "broaden_query",
        should_broaden,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        }
    )
    
    # generate → hallucination_check
    workflow.add_edge("generate", "hallucination_check")
    
    # hallucination_check → conditional (end или generate для retry)
    # Если grounded → end, иначе → generate для регенерации
    workflow.add_conditional_edges(
        "hallucination_check",
        should_regenerate,
        {
            "end": END,
            "generate": "generate",  # retry - регенерировать ответ
        }
    )
    
    # Компилируем граф
    compiled_graph = workflow.compile()
    
    return compiled_graph


# Глобальный экземпляр графа
_rag_graph = None


def get_rag_graph() -> StateGraph:
    """Получить экземпляр RAG графа."""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = create_rag_graph()
    return _rag_graph


def run_rag_pipeline(
    question: str,
    config: Optional[dict] = None,
) -> RAGState:
    """Запустить RAG пайплайн.
    
    Args:
        question: Вопрос пользователя.
        config: Конфигурация для графа (опционально).
        
    Returns:
        Финальное состояние.
    """
    graph = get_rag_graph()
    
    # Создаём начальное состояние
    initial_state = create_initial_state(question)
    
    # Запускаем граф
    result = graph.invoke(
        initial_state,
        config=config or {"configurable": {"thread_id": "default"}},
    )
    
    return result


async def run_rag_pipeline_async(
    question: str,
    config: Optional[dict] = None,
) -> RAGState:
    """Асинхронно запустить RAG пайплайн.
    
    Args:
        question: Вопрос пользователя.
        config: Конфигурация для графа (опционально).
        
    Returns:
        Финальное состояние.
    """
    graph = get_rag_graph()
    
    # Создаём начальное состояние
    initial_state = create_initial_state(question)
    
    # Асинхронно запускаем граф
    result = await graph.ainvoke(
        initial_state,
        config=config or {"configurable": {"thread_id": "default"}},
    )
    
    return result