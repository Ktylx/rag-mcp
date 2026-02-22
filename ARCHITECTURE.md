# Architecture

## Overview

RAG MCP Server is a Model Context Protocol (MCP) server that provides a knowledge base system with local LLM integration. It allows users to index documents, search them using semantic similarity, and ask questions with AI-generated answers.

## System Components

### 1. MCP Server (`src/server.py`)

The MCP server exposes 5 tools:
- `index_folder` - Index documents from a folder
- `ask_question` - Ask questions using RAG pipeline
- `find_relevant_docs` - Find relevant documents without generation
- `summarize_document` - Summarize a single document
- `index_status` - Get index statistics

### 2. Indexer Module (`src/indexer/`)

Responsible for loading and processing documents:

- **DocumentLoader** (`document_loader.py`) - Loads documents from various formats:
  - Text: `.txt`, `.md`, `.rst`
  - Code: `.py`, `.js`, `.ts`
  - Data: `.json`, `.yaml`, `.yml`
  - PDF: `.pdf` (optional)

- **TextSplitter** (`text_splitter.py`) - Splits documents into chunks:
  - Text files: Uses `RecursiveCharacterTextSplitter` with paragraph/sentence separators
  - Code files: Uses language-aware splitting (functions, classes, blocks)

- **ChromaManager** (`chroma_manager.py`) - Manages vector storage:
  - Persistent ChromaDB storage
  - Similarity search with scores
  - Statistics tracking

### 3. RAG Engine (`src/rag/`)

LangGraph-based retrieval-augmented generation pipeline:

- **State** (`state.py`) - Defines the RAGState TypedDict with all pipeline state

- **Nodes** (`nodes.py`) - Graph nodes:
  - `rewrite_query` - Rewrites query for better retrieval
  - `retrieve` - Retrieves documents from vector store
  - `grade_documents` - Grades document relevance
  - `generate` - Generates answer using LLM
  - `hallucination_check` - Checks if answer is grounded
  - `broaden_query` - Expands query when too few results

- **Prompts** (`prompts.py`) - LLM prompt templates

- **Graph** (`graph.py`) - LangGraph workflow definition

### 4. Utilities (`src/utils/`)

- **OllamaClient** (`ollama_client.py`) - Manages LLM and embedding connections

## Data Flow

```
User Query
    ↓
rewrite_query (LLM)
    ↓
retrieve (Vector Search)
    ↓
grade_documents (LLM)
    ↓ (if enough relevant)
generate (LLM)
    ↓
hallucination_check (LLM)
    ↓ (if not grounded, retry)
Return Answer + Sources
```

## Configuration

All configuration is in `src/config.py`:
- Chunk size and overlap
- Ollama URL and models
- RAG parameters (top_k, max retries, etc.)
- Prompt templates

## Docker Compose

The system runs with:
- **ollama** service - Local LLM server
- **server** service - MCP server

## Storage

- ChromaDB data: `./chroma_data/`
- Sample documents: `./sample_docs/`