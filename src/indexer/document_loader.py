"""
Загрузчик документов различных форматов.
Поддерживает: .md, .txt, .rst, .py, .js, .ts, .json, .yaml, .yml, .pdf
"""

import re
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
    JSONLoader,
)
from langchain_community.document_loaders.pdf import PyPDFLoader

# Try to import YAMLLoader - may not be available in all versions
try:
    from langchain_community.document_loaders import YAMLLoader
except ImportError:
    YAMLLoader = None
from langchain_core.documents import Document

from src import config


# Common encodings to try when UTF-8 fails
ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]


def clean_text_content(text: str) -> str:
    """Remove invalid control characters from text.
    
    Removes Unicode control characters (U+0000 to U+001F except tab, newline, carriage return)
    which often cause issues with text processing.
    """
    # Keep tab (\t), newline (\n), and carriage return (\r)
    # Remove all other control characters (U+0000 to U+001F and U+007F to U+009F)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return cleaned


class DocumentLoader:
    """Загрузчик документов различных форматов."""

    # Расширения и соответствующие загрузчики
    LOADERS = {
        ".md": TextLoader,
        ".rst": TextLoader,
        ".txt": TextLoader,
        ".py": PythonLoader,
        ".js": TextLoader,
        ".ts": TextLoader,
        ".json": JSONLoader,
        ".pdf": PyPDFLoader,
    }

    def __init__(self, encoding: str = "utf-8"):
        """Инициализация загрузчика.
        
        Args:
            encoding: Кодировка для текстовых файлов.
        """
        # Добавляем YAML загрузчик если доступен
        if YAMLLoader is not None:
            self.LOADERS[".yaml"] = YAMLLoader
            self.LOADERS[".yml"] = YAMLLoader
        
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
        
        # Проверка для YAML файлов если загрузчик недоступен
        if loader_class is None:
            raise ValueError(f"YAMLLoader not available. Install langchain-community with YAML support.")
        
        # Для текстовых файлов пробуем разные кодировки
        if extension in (".txt", ".rst", ".js", ".ts"):
            docs = self._load_text_with_fallback(file_path, loader_class)
        elif extension == ".json":
            loader = loader_class(str(file_path), jq_schema=".")
            docs = loader.load()
        elif extension == ".yaml":
            loader = loader_class(str(file_path))
            docs = loader.load()
        else:
            loader = loader_class(str(file_path))
            docs = loader.load()
        
        # Добавляем метаданные о файле и очищаем текст
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_extension"] = extension
            # Очищаем текст от некорректных управляющих символов
            doc.page_content = clean_text_content(doc.page_content)
        
        return docs

    def _load_text_with_fallback(
        self, 
        file_path: Path, 
        loader_class
    ) -> List[Document]:
        """Загрузить текстовый файл с пробразом разных кодировок.
        
        Args:
            file_path: Путь к файлу.
            loader_class: Класс загрузчика (TextLoader).
            
        Returns:
            Список документов.
        """
        # Пробуем каждую кодировку из списка
        for encoding in ENCODING_FALLBACKS:
            try:
                loader = loader_class(str(file_path), encoding=encoding)
                docs = loader.load()
                return docs
            except (UnicodeDecodeError, UnicodeError) as e:
                # Пробуем следующую кодировку
                continue
            except Exception as e:
                # Другие ошибки - пробуем следующую кодировку
                continue
        
        # Если ничего не помогло, пробуем прочитать файл вручную и создать Document
        try:
            # Читаем файл в бинарном режиме
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Пробуем декодировать с игнорированием ошибок
            for encoding in ENCODING_FALLBACKS:
                try:
                    text = raw_data.decode(encoding, errors='ignore')
                    text = clean_text_content(text)
                    break
                except Exception:
                    continue
            else:
                # Если ничего не помогло, используем latin-1 с игнорированием ошибок
                text = raw_data.decode('latin-1', errors='ignore')
                text = clean_text_content(text)
            
            from langchain_core.documents import Document
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix.lower()
                }
            )
            return [doc]
        except Exception as e:
            raise ValueError(f"Failed to load file {file_path} with any encoding: {e}")

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