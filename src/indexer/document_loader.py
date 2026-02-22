"""
Загрузчик документов различных форматов.
Поддерживает: .md, .txt, .rst, .py, .js, .ts, .json, .yaml, .yml, .pdf
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
    JSONLoader,
    YAMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

from src import config


class DocumentLoader:
    """Загрузчик документов различных форматов."""

    # Расширения и соответствующие загрузчики
    LOADERS = {
        ".md": UnstructuredMarkdownLoader,
        ".rst": TextLoader,
        ".txt": TextLoader,
        ".py": PythonLoader,
        ".js": TextLoader,
        ".ts": TextLoader,
        ".json": JSONLoader,
        ".yaml": YAMLLoader,
        ".yml": YAMLLoader,
        ".pdf": PyPDFLoader,
    }

    def __init__(self, encoding: str = "utf-8"):
        """Инициализация загрузчика.
        
        Args:
            encoding: Кодировка для текстовых файлов.
        """
        self.encoding = encoding

    def load_file(self, file_path: Path) -> List[Document]:
        """Загрузить один файл.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Список документов (Document).
            
        Raises:
            ValueError: Если формат файла не поддерживается.
        """
        extension = file_path.suffix.lower()
        
        if extension not in self.LOADERS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        loader_class = self.LOADERS[extension]
        
        # Для текстовых файлов используем TextLoader с явной кодировкой
        if extension in (".txt", ".rst", ".js", ".ts"):
            loader = loader_class(str(file_path), encoding=self.encoding)
        elif extension == ".json":
            loader = loader_class(str(file_path), jq_schema=".")
        elif extension == ".yaml":
            loader = loader_class(str(file_path))
        else:
            loader = loader_class(str(file_path))
        
        docs = loader.load()
        
        # Добавляем метаданные о файле
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_extension"] = extension
        
        return docs

    def load_folder(
        self, 
        folder_path: Path, 
        glob_pattern: str = "**/*"
    ) -> List[Document]:
        """Загрузить все файлы из папки.
        
        Args:
            folder_path: Путь к папке.
            glob_pattern: Glob паттерн для фильтрации файлов.
            
        Returns:
            Список всех документов.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")
        
        all_docs = []
        
        for file_path in folder.glob(glob_pattern):
            if not file_path.is_file():
                continue
            
            extension = file_path.suffix.lower()
            if extension not in config.SUPPORTED_EXTENSIONS:
                continue
            
            try:
                docs = self.load_file(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return all_docs


def load_documents(
    path: Path, 
    glob_pattern: Optional[str] = None
) -> List[Document]:
    """Удобная функция для загрузки документов.
    
    Args:
        path: Путь к файлу или папке.
        glob_pattern: Glob паттерн для файлов (если path - папка).
        
    Returns:
        Список документов.
    """
    loader = DocumentLoader()
    
    if path.is_file():
        return loader.load_file(path)
    elif path.is_dir():
        pattern = glob_pattern or config.DEFAULT_GLOB_PATTERN
        return loader.load_folder(path, pattern)
    else:
        raise ValueError(f"Path not found: {path}")