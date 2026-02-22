"""
Клиент для взаимодействия с Ollama.
"""

from typing import Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src import config


# Глобальные экземпляры
_llm: Optional[BaseChatModel] = None
_embeddings: Optional[Embeddings] = None


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Получить экземпляр LLM через Ollama.
    
    Args:
        model: Название модели (по умолчанию из конфига).
        temperature: Температура генерации.
        max_tokens: Максимальное количество токенов.
        
    Returns:
        Экземпляр ChatOllama.
    """
    global _llm
    
    model = model or config.LLM_MODEL
    
    if _llm is None or getattr(_llm, 'model', None) != model:
        _llm = ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.OLLAMA_TIMEOUT,
        )
    
    return _llm


def get_embedding_client(
    model: Optional[str] = None,
) -> Embeddings:
    """Получить клиента для эмбеддингов.
    
    Args:
        model: Название модели (по умолчанию из конфига).
        
    Returns:
        Экземпляр OllamaEmbeddings.
    """
    global _embeddings
    
    if config.USE_OLLAMA_EMBEDDINGS:
        model = model or config.EMBEDDING_MODEL
    else:
        model = "nomic-embed-text"  # Дефолтная модель
    
    if _embeddings is None or getattr(_embeddings, 'model', None) != model:
        _embeddings = OllamaEmbeddings(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
        )
    
    return _embeddings


def reset_llm():
    """Сбросить экземпляры LLM и эмбеддингов."""
    global _llm, _embeddings
    _llm = None
    _embeddings = None