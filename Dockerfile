FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY pyproject.toml .
COPY pytest.ini .
RUN pip install --no-cache-dir -e ".[dev]"

# Копирование исходного кода
COPY src/ ./src/
COPY tests/ ./tests/

# Создание директорий
RUN mkdir -p /app/chroma_data /app/sample_docs

# Умолчательная команда
CMD ["python", "-m", "src.server"]