"""
Менеджер ChromaDB для хранения и поиска векторных представлений документов.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src import config


class ChromaManager:
    """Менеджер ChromaDB для работы с векторным хранилищем."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "rag_documents",
    ):
        """Инициализация менеджера ChromaDB.
        
        Args:
            persist_directory: Директория для сохранения данных.
            collection_name: Имя коллекции.
        """
        self.persist_directory = persist_directory or config.get_chroma_path()
        self.collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._vectorstore: Optional[Chroma] = None
        self._index_stats: Dict[str, Any] = {
            "file_count": 0,
            "chunk_count": 0,
            "last_indexed": None,
            "indexed_files": [],
        }

    def _get_embeddings(self):
        """Получить эмбеддинг-модель."""
        if config.USE_OLLAMA_EMBEDDINGS:
            return OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
        else:
            # Используем дефолтную модель ChromaDB
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=config.OLLAMA_BASE_URL,
            )

    def _get_client(self) -> chromadb.PersistentClient:
        """Получить клиента ChromaDB."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    def _get_vectorstore(self) -> Chroma:
        """Получить векторное хранилище."""
        if self._vectorstore is None:
            embeddings = self._get_embeddings()
            self._vectorstore = Chroma(
                client=self._get_client(),
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vectorstore

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Добавить документы в хранилище.
        
        Args:
            documents: Список документов.
            ids: Список ID для документов (опционально).
            
        Returns:
            Список ID добавленных документов.
        """
        if not documents:
            return []
        
        # Генерируем IDs если не предоставлены
        if ids is None:
            ids = [f"doc_{i}_{hash(doc.page_content) % 100000}" 
                   for i, doc in enumerate(documents)]
        
        vectorstore = self._get_vectorstore()
        
        # Добавляем документы
        vectorstore.add_documents(documents=documents, ids=ids)
        
        # Обновляем статистику
        self._update_stats(documents)
        
        return ids

    def _update_stats(self, documents: List[Document]):
        """Обновить статистику индекса."""
        # Собираем уникальные файлы
        files = set()
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source:
                files.add(source)
        
        # Обновляем счетчики
        self._index_stats["file_count"] += len(files)
        self._index_stats["chunk_count"] += len(documents)
        self._index_stats["last_indexed"] = datetime.now().isoformat()
        
        for file in files:
            if file not in self._index_stats["indexed_files"]:
                self._index_stats["indexed_files"].append(file)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Поиск похожих документов.
        
        Args:
            query: Запрос для поиска.
            k: Количество результатов.
            filter: Фильтр по метаданным.
            
        Returns:
            Список найденных документов.
        """
        vectorstore = self._get_vectorstore()
        return vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Поиск с оценкой релевантности.
        
        Args:
            query: Запрос для поиска.
            k: Количество результатов.
            filter: Фильтр по метаданным.
            
        Returns:
            Список (документ, оценка) пар.
        """
        vectorstore = self._get_vectorstore()
        return vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def delete_collection(self):
        """Удалить коллекцию."""
        client = self._get_client()
        client.delete_collection(name=self.collection_name)
        self._vectorstore = None
        
        # Сбрасываем статистику
        self._index_stats = {
            "file_count": 0,
            "chunk_count": 0,
            "last_indexed": None,
            "indexed_files": [],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику индекса.
        
        Returns:
            Словарь со статистикой.
        """
        try:
            vectorstore = self._get_vectorstore()
            count = vectorstore._collection.count()
            
            return {
                "chunk_count": count,
                "file_count": self._index_stats["file_count"],
                "indexed_files": self._index_stats["indexed_files"],
                "last_indexed": self._index_stats["last_indexed"],
                "collection_name": self.collection_name,
            }
        except Exception as e:
            return {
                "error": str(e),
                **self._index_stats,
            }

    def reset(self):
        """Сбросить хранилище."""
        self.delete_collection()
        # Удаляем директорию с данными
        if self.persist_directory.exists():
            import shutil
            shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр менеджера
_global_chroma_manager: Optional[ChromaManager] = None


def get_chroma_manager() -> ChromaManager:
    """Получить глобальный экземпляр ChromaManager."""
    global _global_chroma_manager
    if _global_chroma_manager is None:
        _global_chroma_manager = ChromaManager()
    return _global_chroma_manager


def reset_chroma_manager():
    """Сбросить глобальный менеджер."""
    global _global_chroma_manager
    _global_chroma_manager = None