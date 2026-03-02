"""
MCP сервер для RAG системы.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP

from src import config
from src.indexer.document_loader import load_documents
from src.indexer.text_splitter import split_documents
from src.indexer.chroma_manager import get_chroma_manager
from src.rag.graph import run_rag_pipeline
from src.rag.prompts import SUMMARIZE_PROMPT
from src.utils.ollama_client import get_llm

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаём MCP сервер
mcp = FastMCP("RAG Knowledge Base")


@mcp.tool()
def index_folder(
    folder_path: str,
    glob_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Индексировать папку с документами.
    
    Сканирует файлы, разбивает на чанки, создаёт эмбеддинги и сохраняет в ChromaDB.
    
    Args:
        folder_path: Путь к папке с документами.
        glob_pattern: Glob паттерн для фильтрации файлов (опционально).
        
    Returns:
        Словарь с результатами индексации.
    """
    path = Path(folder_path)
    
    if not path.exists():
        return {
            "success": False,
            "error": f"Path not found: {folder_path}",
        }
    
    try:
        # Загружаем документы
        logger.info(f"Loading documents from {folder_path}")
        documents = load_documents(path, glob_pattern)
        
        if not documents:
            return {
                "success": False,
                "error": "No documents found",
                "files_indexed": 0,
                "chunks_created": 0,
            }
        
        # Разбиваем на чанки
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = split_documents(documents)
        
        # Сохраняем в ChromaDB
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
        chroma = get_chroma_manager()
        chroma.add_documents(chunks)
        
        # Получаем статистику
        stats = chroma.get_stats()
        
        return {
            "success": True,
            "files_indexed": stats.get("file_count", len(documents)),
            "chunks_created": len(chunks),
            "total_chunks": stats.get("chunk_count", len(chunks)),
            "message": f"Successfully indexed {len(chunks)} chunks from {len(documents)} documents",
        }
        
    except Exception as e:
        logger.error(f"Error indexing folder: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def ask_question(question: str) -> Dict[str, Any]:
    """Задать вопрос на основе индексированных документов.
    
    Запускает полный RAG пайплайн (LangGraph) для генерации ответа.
    
    Args:
        question: Вопрос пользователя.
        
    Returns:
        Словарь с ответом и источниками.
    """
    try:
        # Запускаем RAG пайплайн
        logger.info(f"Processing question: {question}")
        result = run_rag_pipeline(question)
        
        return {
            "success": True,
            "answer": result.get("generation", ""),
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query", ""),
            "documents_used": len(result.get("graded_documents", [])),
        }
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {
            "success": False,
            "error": str(e),
            "answer": "Извините, произошла ошибка при обработке вашего вопроса.",
        }


@mcp.tool()
def find_relevant_docs(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Найти релевантные документы.
    
    Возвращает ранжированные чанки без генерации ответа.
    
    Args:
        query: Запрос для поиска.
        top_k: Количество результатов (по умолчанию из конфига).
        
    Returns:
        Словарь с найденными документами.
    """
    k = top_k or config.DEFAULT_TOP_K
    
    try:
        chroma = get_chroma_manager()
        
        # Поиск с оценками
        results = chroma.similarity_search_with_score(query, k=k)
        
        documents = []
        for doc, score in results:
            documents.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            })
        
        return {
            "success": True,
            "query": query,
            "documents": documents,
            "count": len(documents),
        }
        
    except Exception as e:
        logger.error(f"Error finding relevant docs: {e}")
        return {
            "success": False,
            "error": str(e),
            "documents": [],
        }


@mcp.tool()
def summarize_document(file_path: str) -> Dict[str, Any]:
    """Создать саммари документа.
    
    Разбивает документ на чанки и генерирует саммари.
    
    Args:
        file_path: Путь к файлу.
        
    Returns:
        Словарь с саммари.
    """
    path = Path(file_path)
    
    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }
    
    try:
        # Загружаем документ
        documents = load_documents(path)
        
        if not documents:
            return {
                "success": False,
                "error": "Could not load document",
            }
        
        # Разбиваем на чанки
        chunks = split_documents(documents)
        
        if not chunks:
            return {
                "success": False,
                "error": "Could not split document",
            }
        
        # Объединяем чанки для саммари (с ограничением длины)
        combined_text = "\n\n".join(
            chunk.page_content for chunk in chunks[:10]  # Ограничиваем 10 чанками
        )
        
        # Генерируем саммари
        llm = get_llm()
        prompt = SUMMARIZE_PROMPT.format(document=combined_text[:3000])
        
        response = llm.invoke(prompt)
        summary = response.content.strip()
        
        return {
            "success": True,
            "file_name": path.name,
            "summary": summary,
            "chunks_processed": len(chunks),
        }
        
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def index_status() -> Dict[str, Any]:
    """Получить статистику индекса.
    
    Возвращает количество файлов, чанков и время последней индексации.
    
    Returns:
        Словарь со статистикой.
    """
    try:
        chroma = get_chroma_manager()
        stats = chroma.get_stats()
        
        return {
            "success": True,
            "file_count": stats.get("file_count", 0),
            "chunk_count": stats.get("chunk_count", 0),
            "last_indexed": stats.get("last_indexed", None),
            "indexed_files": stats.get("indexed_files", []),
            "collection_name": stats.get("collection_name", "rag_documents"),
        }
        
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_count": 0,
            "chunk_count": 0,
        }


def main():
    """Запуск MCP сервера."""
    import sys
    
    # Проверяем аргументы командной строки
    transport = "stdio"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--http":
            transport = "http"
    
    logger.info(f"Starting RAG MCP Server with {transport} transport...")
    try:
        if transport == "http":
            mcp.run(transport="http", host="0.0.0.0", port=8000)
        else:
            mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()