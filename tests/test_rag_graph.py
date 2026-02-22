"""
Unit tests for the RAG graph with mocked LLM.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from src.rag.state import RAGState, create_initial_state
from src.rag.nodes import (
    rewrite_query_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    hallucination_check_node,
    broaden_query_node,
    should_continue_grade,
    should_regenerate,
    should_broaden,
)
from src.rag.graph import create_rag_graph, run_rag_pipeline


class TestRAGState:
    """Tests for RAG state."""
    
    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state("What is Python?")
        
        assert state["question"] == "What is Python?"
        assert state["rewritten_query"] == "What is Python?"
        assert state["documents"] == []
        assert state["graded_documents"] == []
        assert state["generation"] == ""
        assert state["sources"] == []
        assert state["retry_count"] == 0
        assert state["broaden_count"] == 0
        assert state["is_grounded"] == False
        assert state["error"] is None


class TestRAGNodes:
    """Tests for RAG nodes."""
    
    @patch("src.rag.nodes.get_llm")
    def test_rewrite_query_node(self, mock_get_llm):
        """Test query rewriting node."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="What is Python programming?")
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        result = rewrite_query_node(state)
        
        assert result["rewritten_query"] == "What is Python programming?"
        assert result["question"] == "What is Python?"
    
    @patch("src.rag.nodes.get_chroma_manager")
    def test_retrieve_node(self, mock_get_chroma):
        """Test retrieve node."""
        mock_chroma = Mock()
        mock_chroma.similarity_search.return_value = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "test.py"}
            )
        ]
        mock_get_chroma.return_value = mock_chroma
        
        state = create_initial_state("What is Python?")
        state["rewritten_query"] = "Python programming"
        
        result = retrieve_node(state)
        
        assert len(result["documents"]) == 1
        assert "test.py" in result["sources"]
    
    @patch("src.rag.nodes.get_llm")
    @patch("src.rag.nodes.get_chroma_manager")
    def test_grade_documents_node_all_relevant(self, mock_get_chroma, mock_get_llm):
        """Test grading documents - all relevant."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="yes")
        mock_get_llm.return_value = mock_llm
        
        mock_chroma = Mock()
        mock_chroma.similarity_search.return_value = [
            Document(page_content="Python is great", metadata={"source": "test1.txt"}),
            Document(page_content="Python is popular", metadata={"source": "test2.txt"}),
        ]
        mock_get_chroma.return_value = mock_chroma
        
        state = create_initial_state("What is Python?")
        state["documents"] = [
            Document(page_content="Python is great", metadata={"source": "test1.txt"}),
            Document(page_content="Python is popular", metadata={"source": "test2.txt"}),
        ]
        
        result = grade_documents_node(state)
        
        assert len(result["graded_documents"]) == 2
    
    @patch("src.rag.nodes.get_llm")
    def test_generate_node_with_docs(self, mock_get_llm):
        """Test generate node with documents."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="Python is a high-level programming language."
        )
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        state["graded_documents"] = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "test.py"}
            )
        ]
        state["sources"] = ["test.py"]
        
        result = generate_node(state)
        
        assert result["generation"] == "Python is a high-level programming language."
    
    def test_generate_node_no_docs(self):
        """Test generate node without documents."""
        state = create_initial_state("What is Python?")
        state["graded_documents"] = []
        state["sources"] = []
        
        result = generate_node(state)
        
        assert "не нашёл" in result["generation"].lower()
        assert result["is_grounded"] == False
    
    @patch("src.rag.nodes.get_llm")
    def test_hallucination_check_grounded(self, mock_get_llm):
        """Test hallucination check - grounded."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="yes")
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        state["generation"] = "Python is a programming language."
        state["graded_documents"] = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "test.py"}
            )
        ]
        
        result = hallucination_check_node(state)
        
        assert result["is_grounded"] == True
    
    @patch("src.rag.nodes.get_llm")
    def test_hallucination_check_not_grounded(self, mock_get_llm):
        """Test hallucination check - not grounded."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="no")
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        state["generation"] = "Python was invented in 1950."
        state["graded_documents"] = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "test.py"}
            )
        ]
        
        result = hallucination_check_node(state)
        
        assert result["is_grounded"] == False
    
    @patch("src.rag.nodes.get_llm")
    def test_broaden_query_node(self, mock_get_llm):
        """Test broaden query node."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="Python programming language features"
        )
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        state["documents"] = [Document(page_content="test", metadata={})]
        state["broaden_count"] = 0
        
        result = broaden_query_node(state)
        
        assert result["rewritten_query"] == "Python programming language features"
        assert result["broaden_count"] == 1
    
    @patch("src.rag.nodes.get_llm")
    def test_broaden_query_max_loops(self, mock_get_llm):
        """Test broaden query - max loops reached."""
        from src import config
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        state = create_initial_state("What is Python?")
        state["broaden_count"] = config.MAX_BROADEN_LOOPS
        
        result = broaden_query_node(state)
        
        # Should not call LLM when max loops reached
        mock_llm.invoke.assert_not_called()


class TestConditionalEdges:
    """Tests for conditional edge functions."""
    
    def test_should_continue_grade_enough_docs(self):
        """Test should_continue_grade with enough documents."""
        state = create_initial_state("test")
        state["graded_documents"] = [
            Document(page_content="test", metadata={}),
            Document(page_content="test", metadata={}),
            Document(page_content="test", metadata={}),
        ]
        
        result = should_continue_grade(state)
        
        assert result == "generate"
    
    def test_should_continue_grade_few_docs(self):
        """Test should_continue_grade with few documents."""
        state = create_initial_state("test")
        state["graded_documents"] = [
            Document(page_content="test", metadata={})
        ]
        
        result = should_continue_grade(state)
        
        assert result == "broaden_query"
    
    def test_should_regenerate_grounded(self):
        """Test should_regenerate when grounded."""
        state = create_initial_state("test")
        state["is_grounded"] = True
        state["retry_count"] = 0
        
        result = should_regenerate(state)
        
        assert result == "end"
    
    def test_should_regenerate_not_grounded(self):
        """Test should_regenerate when not grounded."""
        state = create_initial_state("test")
        state["is_grounded"] = False
        state["retry_count"] = 0
        
        from src import config
        
        result = should_regenerate(state)
        
        assert result == "generate"
    
    def test_should_regenerate_max_retries(self):
        """Test should_regenerate with max retries."""
        from src import config
        
        state = create_initial_state("test")
        state["is_grounded"] = False
        state["retry_count"] = config.MAX_REGENERATE_RETRIES + 1
        
        result = should_regenerate(state)
        
        assert result == "end"
    
    def test_should_broaden_continue(self):
        """Test should_broaden to continue."""
        from src import config
        
        state = create_initial_state("test")
        state["broaden_count"] = 0
        
        result = should_broaden(state)
        
        assert result == "retrieve"
    
    def test_should_broaden_max_loops(self):
        """Test should_broaden with max loops."""
        from src import config
        
        state = create_initial_state("test")
        state["broaden_count"] = config.MAX_BROADEN_LOOPS
        
        result = should_broaden(state)
        
        assert result == "generate"


class TestRAGGraph:
    """Tests for RAG graph."""
    
    def test_create_rag_graph(self):
        """Test creating RAG graph."""
        graph = create_rag_graph()
        
        assert graph is not None
    
    @patch("src.rag.graph.run_rag_pipeline")
    def test_run_rag_pipeline(self, mock_run):
        """Test running RAG pipeline."""
        mock_run.return_value = {
            "question": "test",
            "generation": "test answer",
            "sources": ["test.py"],
        }
        
        result = run_rag_pipeline("test")
        
        assert "generation" in result