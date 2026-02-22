"""
LangGraph граф для Corrective RAG пайплайна.
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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

# Инициализируем граф
workflow = StateGraph(RAGState)


def create_rag_graph() -> StateGraph:
    """Создать LangGraph граф для RAG пайплайна.
    
    Returns:
        Скомпилированный граф.
    """
    # Добавляем узлы
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("broaden_query", broaden_query_node)
    
    # Устанавливаем начальный узел
    workflow.set_entry_point("rewrite_query")
    
    # Добавляем рёбра
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Условные переходы после оценки документов
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue_grade,
        {
            "generate": "generate",
            "broaden_query": "broaden_query",
        },
    )
    
    # Цикл расширения запроса
    workflow.add_conditional_edges(
        "broaden_query",
        should_broaden,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("generate", "hallucination_check")
    
    # Условный переход после проверки на галлюцинации
    workflow.add_conditional_edges(
        "hallucination_check",
        should_regenerate,
        {
            "end": END,
            "generate": "generate",
        },
    )
    
    # Компилируем граф
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
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