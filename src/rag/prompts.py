"""
Prompts for RAG graph.
"""

from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src import config


# Prompt for query rewriting
QUERY_REWRITE_PROMPT = PromptTemplate.from_template(
    """Rewrite the following question to make it more effective for semantic search in a knowledge base. Keep the essence of the question but rephrase it with different words.

Question: {question}

Rewritten question:"""
)


# Prompt for grading document relevance
GRADE_DOCUMENTS_PROMPT = PromptTemplate.from_template(
    """Determine if the document is relevant to the user's query.

Query: {question}

Document: {document}

Answer with ONLY one word: "yes" if the document is relevant, "no" if it is not relevant."""
)


# Prompt for answer generation
def create_generation_prompt(documents: List[Document], sources: List[str]) -> str:
    """Create prompt for answer generation."""
    docs_text = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )
    sources_text = ", ".join(sources)
    
    return f"""You are an assistant that answers questions based on the provided documents.

Use ONLY information from the provided documents. If the information is insufficient, honestly say so.

Query: {{question}}

Documents:
{docs_text}

Answer the query based on the documents. After your answer, list the sources in this format:
Sources: {sources_text}"""


GENERATION_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are an assistant that answers questions based on the provided documents.

Use ONLY information from the provided documents. If the information is insufficient, honestly say so.

Query: {question}

Documents:
{documents}

Answer the query based on the documents. After your answer, list the sources in this format:
Sources: {sources}"""
)


# Prompt for hallucination check
HALLUCINATION_CHECK_PROMPT = PromptTemplate.from_template(
    """Check if the generated answer is based on the provided documents.

Query: {question}

Generated answer: {generation}

Documents:
{documents}

Answer with ONLY one word: "yes" if the answer is based on the documents, "no" if it contains information not from the documents."""
)


# Prompt for broadening query
BROADEN_QUERY_PROMPT = PromptTemplate.from_template(
    """Broaden and rewrite the query to find more relevant documents.

Current query: {question}
Documents found: {doc_count}

Broadened query:"""
)


# Prompt for summarization
SUMMARIZE_PROMPT = PromptTemplate.from_template(
    """Create a brief summary of the provided document.

Document: {document}

Summary:"""
)


def format_documents_for_prompt(documents: List[Document]) -> str:
    """Отформатировать документы для промпта.
    
    Args:
        documents: Список документов.
        
    Returns:
        Отформатированная строка.
    """
    return "\n\n---\n\n".join(
        f"Документ {i+1} (источник: {doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )