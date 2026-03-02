"""
LangGraph граф для RAG пайплайна (упрощённая версия).
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
)

logger = logging.getLogger(__name__)

def create_rag_graph() -> StateGraph:
    """Создать LangGraph граф для RAG пайплайна.
    
    Returns:
        Скомпилированный граф.
    """
    # Создаём новый граф при каждом вызове
    workflow = StateGraph(RAGState)
    
    # Добавляем узлы (упрощённая схема без циклов)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    
    # Устанавливаем начальный узел
    workflow.set_entry_point("rewrite_query")
    
    # Линейный поток без условных переходов
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    
    # Компилируем граф (без checkpointer для упрощения)
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