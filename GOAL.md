# Project Report

## Overview

This project implements an MCP (Model Context Protocol) server for a RAG (Retrieval-Augmented Generation) knowledge base system using local LLM through Ollama and LangGraph for orchestration.

## Development Process

### Phase 1: Project Setup

- Created project structure with pyproject.toml, Dockerfile, and docker-compose.yml
- Set up configuration in `src/config.py` with all customizable parameters

### Phase 2: Indexer Implementation

- **DocumentLoader**: Implemented support for multiple file formats
  - Text files (.md, .txt, .rst)
  - Code files (.py, .js, .ts)
  - Structured data (.json, .yaml)
  - PDF support (optional)

- **TextSplitter**: Implemented two types of splitting
  - RecursiveCharacterTextSplitter for text (paragraphs → lines → sentences)
  - Language-aware splitter for code (functions, classes, blocks)

- **ChromaManager**: Implemented vector storage with:
  - Persistent storage
  - Similarity search with scores
  - Statistics tracking

### Phase 3: RAG Engine

Implemented Corrective RAG using LangGraph:

1. **Query Rewrite** - Rewrites user query for better retrieval
2. **Retrieve** - Semantic search in ChromaDB
3. **Grade Documents** - LLM-based relevance grading
4. **Generate** - Answer generation with sources
5. **Hallucination Check** - Verify answer is grounded in documents
6. **Broaden Query** - Expand query if too few results

### Phase 4: MCP Server

Created 5 MCP tools:
- `index_folder` - Index documents from folder
- `ask_question` - RAG pipeline question answering
- `find_relevant_docs` - Semantic search only
- `summarize_document` - Document summarization
- `index_status` - Index statistics

### Phase 5: Testing

Created 10+ tests:
- Unit tests for indexer module
- Unit tests for RAG graph with mocked LLM
- E2E tests for MCP tools

### Phase 6: Documentation

- ARCHITECTURE.md - System architecture
- README.md - Usage instructions
- REPORT.md - This file

## Challenges and Solutions

### Challenge 1: FastMCP API
**Problem**: FastMCP API may differ from standard MCP implementation.
**Solution**: Used FastMCP decorators (@mcp.tool()) for tool registration.

### Challenge 2: LangGraph State Management
**Problem**: Managing state across graph nodes with proper typing.
**Solution**: Used TypedDict for RAGState with clear field definitions.

### Challenge 3: Document Loading
**Problem**: Different file formats require different loaders.
**Solution**: Created a mapping of extensions to loaders in DocumentLoader class.

### Challenge 4: Code Splitting
**Problem**: Code files need special handling for meaningful chunks.
**Solution**: Used language-aware splitters from LangChain.

## Prompt Engineering

### Query Rewrite Prompt
```
Переформулируй следующий запрос так, чтобы он был более эффективным для поиска в базе знаний.
```

### Grade Documents Prompt
```
Определи, релевантен ли каждый документ к запросу пользователя.
Ответь ТОЛЬКО одним словом: "yes" или "no"
```

### Generation Prompt
```
Ты - ассистент, который отвечает на вопросы на основе предоставленных документов.
Используй ТОЛЬКО информацию из предоставленных документов.
```

### Hallucination Check Prompt
```
Проверь, основан ли сгенерированный ответ на предоставленных документах.
Ответь ТОЛЬКО одним словом: "yes" или "no"
```

## Future Improvements

1. **PDF Support**: Add full PDF parsing support
2. **External Embeddings**: Support for Ollama embeddings (nomic-embed-text)
3. **Caching**: Add response caching for common queries
4. **Web UI**: Simple web interface for testing
5. **Metrics**: Add metrics and monitoring

## Conclusion

The project successfully implements all required features:
- 5 MCP tools working
- Corrective RAG with LangGraph
- Document indexing with multiple formats
- 10+ tests
- Docker Compose setup
- Complete documentation