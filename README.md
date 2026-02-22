# RAG MCP Server

MCP-сервер с локальной LLM и LangGraph для RAG (Retrieval-Augmented Generation).

## Возможности

- **Индексация документов** — поддержка `.md`, `.txt`, `.rst`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`
- **Векторный поиск** — семантический поиск через ChromaDB
- **RAG-пайплайн** — LangGraph с Corrective RAG (переформулирование запроса, оценка релевантности, проверка на галлюцинации)
- **Локальная LLM** — работа с Ollama без платных API

## Быстрый старт

### Docker Compose

```bash
# Запуск всех сервисов
docker compose up

# Ожидание загрузки модели (при первом запуске)
docker compose logs -f ollama
```

### Локальный запуск

```bash
# Установка зависимостей
pip install -e ".[dev]"

# Запуск Ollama (отдельным терминалом)
ollama serve
ollama pull phi3:mini

# Запуск MCP сервера
python -m src.server
```

## Использование

### Подключение к IDE

Подключите MCP-сервер к вашей IDE (VS Code с расширением MCP, Claude Desktop и т.д.):

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "python",
      "args": ["-m", "src.server"]
    }
  }
}
```

### Доступные инструменты

1. **index_folder** — индексировать папку с документами
```python
index_folder("./sample_docs")
```

2. **ask_question** — задать вопрос
```python
ask_question("Как оформлять docstrings в Python?")
```

3. **find_relevant_docs** — найти релевантные документы
```python
find_relevant_docs("Python docstrings", top_k=5)
```

4. **summarize_document** — создать саммари документа
```python
summarize_document("./sample_docs/api_reference.md")
```

5. **index_status** — получить статистику индекса
```python
index_status()
```

## Тесты

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=src
```

## Структура проекта

```
rag-mcp/
├── src/
│   ├── config.py           # Конфигурация
│   ├── server.py           # MCP сервер
│   ├── indexer/            # Индексация документов
│   ├── rag/                # RAG движок (LangGraph)
│   └── utils/              # Утилиты
├── sample_docs/            # Демо-документы
├── tests/                  # Тесты
├── docker-compose.yml      # Docker Compose
└── Dockerfile
```

## Конфигурация

Параметры находятся в `src/config.py`:

- `CHUNK_SIZE` — размер чанка (по умолчанию 1000)
- `CHUNK_OVERLAP` — перекрытие чанков (по умолчанию 200)
- `OLLAMA_BASE_URL` — URL Ollama (по умолчанию http://localhost:11434)
- `LLM_MODEL` — модель для генерации (по умолчанию phi3:mini)
- `DEFAULT_TOP_K` — количество результатов поиска (по умолчанию 5)

## Требования

- Python 3.11+
- Ollama (локально или Docker)
- Docker и Docker Compose (для контейнеризации)