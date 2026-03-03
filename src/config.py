"""
Конфигурация RAG MCP сервера.
Все параметры вынесены в этот файл для удобства настройки.
"""

from pathlib import Path
from typing import Optional

# ===================
# Пути
# ===================
# Директория для хранения ChromaDB данных
CHROMA_DATA_DIR = Path("./chroma_data")

# Директория с демо-документами
SAMPLE_DOCS_DIR = Path("./sample_docs")

# ===================
# Параметры индексации
# ===================
# Размер чанка для текстовых файлов
CHUNK_SIZE = 1000

# Перекрытие чанков (для сохранения контекста)
CHUNK_OVERLAP = 200

# Glob паттерн для сканирования файлов по умолчанию
DEFAULT_GLOB_PATTERN = "**/*"

# Поддерживаемые расширения файлов
SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".rst",  # Текстовые
    ".py", ".js", ".ts",    # Код
    ".json", ".yaml", ".yml",  # Структурированные
    ".pdf",                  # PDF (опционально)
}

# ===================
# Параметры LLM (Ollama)
# ===================
# URL Ollama сервера (измените на нужный адрес)
# Примеры:
#   "http://localhost:11434" - локальный сервер
#   "http://192.168.1.100:11434" - удалённый сервер
#   "http://ollama:11434" - Docker Compose внутри сети
OLLAMA_BASE_URL = "http://localhost:11434"

# Название модели для генерации ответов
LLM_MODEL = "phi3:mini"

# Название модели для эмбеддингов (если используется Ollama)
EMBEDDING_MODEL = "nomic-embed-text"

# Использовать Ollama для эмбеддингов (иначе - ChromaDB default)
USE_OLLAMA_EMBEDDINGS = True

# Таймаут для запросов к Ollama (секунды)
OLLAMA_TIMEOUT = 120

# ===================
# Параметры RAG
# ===================
# Количество документов для извлечения при поиске
DEFAULT_TOP_K = 30

# Максимальное количество попыток регенерации при галлюцинациях
MAX_REGENERATE_RETRIES = 1

# Максимальное количество циклов при расширении запроса
MAX_BROADEN_LOOPS = 2

# Максимальное общее количество итераций (защита от бесконечного цикла)
MAX_TOTAL_ITERATIONS = 10

# Минимальное количество релевантных документов для генерации ответа
MIN_RELEVANT_DOCS = 2

# ===================
# Параметры сервера
# ===================
# Host для MCP сервера
MCP_HOST = "0.0.0.0"

# Порт для MCP сервера
MCP_PORT = 8000

# ===================
# Prompt шаблоны
# ===================

# Prompt for query rewriting (keep in English to match document language)
QUERY_REWRITE_PROMPT = """Rewrite the following question to make it more effective for semantic search in a knowledge base. Keep the essence of the question but rephrase it with different words.

Question: {question}

Rewritten question:"""

# Prompt for grading document relevance
GRADE_DOCUMENTS_PROMPT = """Determine if the document is relevant to the user's query.

Query: {question}

Document: {document}

Answer with ONLY one word: "yes" if the document is relevant, "no" if it is not relevant."""

# Prompt for answer generation
GENERATION_PROMPT = """You are an assistant that answers questions based on the provided documents.

Use ONLY information from the provided documents. If the information is insufficient, honestly say so.

Query: {question}

Documents:
{documents}

Answer the query based on the documents. After your answer, list the sources in this format:
Sources: {sources}"""

# Prompt for hallucination check
HALLUCINATION_CHECK_PROMPT = """Check if the generated answer is based on the provided documents.

Query: {question}

Generated answer: {generation}

Documents:
{documents}

Answer with ONLY one word: "yes" if the answer is based on the documents, "no" if it contains information not from the documents."""

# Prompt for broadening query
BROADEN_QUERY_PROMPT = """Broaden and rewrite the query to find more relevant documents.

Current query: {question}
Documents found: {doc_count}

Broadened query:"""

# Prompt for summarization
SUMMARIZE_PROMPT = """Create a brief summary of the provided document.

Document: {document}

Summary:"""


def get_chroma_path() -> Path:
    """Получить путь для хранения данных ChromaDB."""
    CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return CHROMA_DATA_DIR


def get_ollama_url() -> str:
    """Получить URL Ollama сервера."""
    return OLLAMA_BASE_URL


def get_llm_model() -> str:
    """Получить название LLM модели."""
    return LLM_MODEL


def get_embedding_model() -> str:
    """Получить название модели для эмбеддингов."""
    return EMBEDDING_MODEL