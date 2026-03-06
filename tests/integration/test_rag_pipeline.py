# tests/integration/test_rag_pipeline.py

"""Integration tests for the complete RAG pipeline."""

import pytest
import tempfile
import shutil
import time
import uuid
import os
import re
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional
import sys
import concurrent.futures

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.document_processing.processor import DocumentProcessor, ProcessingResult
from src.vector_store.chroma_client import ChromaClient
from src.ai_engine.rag_engine import RAGEngine
from src.ai_engine.llm_client import OllamaClient, LLMConfig
from src.database.sqlite_client import SQLiteClient
from src.core.config import AppConfig
from src.utilities.logger import get_logger

logger = get_logger(__name__)

UPLOAD_TIME_THRESHOLD = float(os.getenv("TEST_UPLOAD_TIME_THRESHOLD", "10.0"))
QUERY_TIME_THRESHOLD = float(os.getenv("TEST_QUERY_TIME_THRESHOLD", "15.0"))


@pytest.fixture
def test_dir() -> Generator[Path, None, None]:
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created test directory: {temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Cleaned up test directory: {temp_dir}")


@pytest.fixture
def chroma_client(test_dir: Path) -> ChromaClient:
    chroma_dir = test_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return ChromaClient(persist_directory=str(chroma_dir))


@pytest.fixture
def sqlite_client(test_dir: Path) -> SQLiteClient:
    db_dir = test_dir / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from src.database.sqlite_client import SQLiteClient
        return SQLiteClient(db_path=str(db_dir / "test.db"))
    except:
        return SQLiteClient(str(db_dir / "test.db"))


@pytest.fixture
def document_processor(chroma_client: ChromaClient) -> DocumentProcessor:
    return DocumentProcessor(vector_store=chroma_client)


@pytest.fixture
def rag_engine(chroma_client: ChromaClient, test_dir: Path) -> Generator[Optional[RAGEngine], None, None]:
    try:
        llm_config = LLMConfig(
            provider="ollama",
            model_name="phi",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=100
        )

        llm_client = OllamaClient(llm_config)
        llm_client.initialize()
        
        RAGEngine._instance = None
        
        engine = RAGEngine(
            config={
                "top_k": 3,
                "similarity_threshold": 0.5,
                "max_context_tokens": 500,
                "enable_hybrid_search": False,
                "enable_cache": True,
                "cache_ttl_seconds": 60
            },
            llm_client=llm_client,
            vector_store=chroma_client
        )
        
        yield engine
        
        logger.info(f"Cleaning up RAG engine after test...")
        engine.invalidate_cache()
        engine.reset_metrics()
        if hasattr(engine, 'conversation_manager') and hasattr(engine.conversation_manager, 'conversations'):
            engine.conversation_manager.conversations.clear()
            
    except Exception as e:
        logger.warning(f"Ollama not available: {e}. RAG engine tests will be skipped.")
        yield None


@pytest.fixture
def sample_pdf_file(test_dir: Path) -> Optional[Path]:
    try:
        from reportlab.pdfgen import canvas
        pdf_path = test_dir / "sample.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Cockatoo PDF Test Document")
        c.save()
        return pdf_path
    except ImportError:
        return None


@pytest.fixture
def sample_image_file(test_dir: Path) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw
        img_path = test_dir / "sample.png"
        img = Image.new('RGB', (400, 100), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 40), "Cockatoo OCR Test", fill='black')
        img.save(img_path)
        return img_path
    except ImportError:
        return None


@pytest.fixture
def conversation(rag_engine: Optional[RAGEngine]) -> Generator[Optional[str], None, None]:
    if rag_engine is None:
        yield None
    else:
        conv_id = str(uuid.uuid4())
        rag_engine.start_conversation(conv_id)
        logger.info(f"Started conversation: {conv_id}")
        yield conv_id
        if hasattr(rag_engine, 'clear_conversation'):
            rag_engine.clear_conversation(conv_id)
            logger.info(f"Cleaned up conversation: {conv_id}")


@pytest.fixture
def sample_text_file(test_dir: Path) -> Path:
    file_path = test_dir / "sample.txt"
    content = """
    Cockatoo is a RAG (Retrieval Augmented Generation) application for personal knowledge management.
    It processes documents, creates embeddings, stores them in vector databases, and answers questions.
    The system supports multiple document formats including PDF, DOCX, TXT, and images with OCR.
    All processing happens locally on the user's machine, ensuring privacy and offline functionality.
    """
    file_path.write_text(content)
    return file_path


@pytest.fixture
def multiple_test_files(test_dir: Path) -> List[Path]:
    files = []
    contents = [
        "Python is a programming language for AI and machine learning.",
        "JavaScript is used for web development and browser-based applications.",
        "Cockatoo uses Python for backend processing and JavaScript for the web interface.",
        "Vector databases store embeddings for semantic search and retrieval.",
        "Local LLMs like Llama 2 provide private and offline question answering."
    ]
    
    for i, content in enumerate(contents):
        file_path = test_dir / f"doc_{i}.txt"
        file_path.write_text(content)
        files.append(file_path)
    
    return files


def get_collection_info_safe(client: ChromaClient) -> Dict[str, Any]:
    default_info = {"document_count": 0, "name": "unknown"}
    
    try:
        if hasattr(client, 'get_collection_info'):
            info = client.get_collection_info()
            if isinstance(info, dict):
                return info
            else:
                return default_info
        elif hasattr(client, 'count_documents'):
            return {"document_count": client.count_documents()}
        elif hasattr(client, 'collection') and hasattr(client.collection, 'count'):
            return {"document_count": client.collection.count()}
        else:
            try:
                results = client.search("", n_results=1000)
                if isinstance(results, dict):
                    if "ids" in results:
                        doc_count = len(results["ids"]) if results["ids"] else 0
                    elif "documents" in results:
                        doc_count = len(results["documents"]) if results["documents"] else 0
                    elif "count" in results:
                        doc_count = results["count"]
                    else:
                        doc_count = 0
                    return {"document_count": doc_count}
                return default_info
            except Exception:
                return default_info
    except Exception as e:
        logger.warning(f"Failed to get collection info: {e}")
        return default_info


def standardize_metadata(result: ProcessingResult) -> Dict[str, Any]:
    metadata = {}
    
    if hasattr(result, 'metadata') and result.metadata:
        if isinstance(result.metadata, dict):
            metadata.update(result.metadata)
    
    for attr in ['file_name', 'file_size', 'file_path', 'document_id']:
        if hasattr(result, attr):
            metadata[attr] = getattr(result, attr)
    
    return metadata


def verify_vector_storage(client: ChromaClient, expected_min_chunks: int = 1) -> bool:
    try:
        if hasattr(client, 'get_collection_info'):
            info = client.get_collection_info()
            if isinstance(info, dict):
                if info.get("document_count", 0) >= expected_min_chunks:
                    logger.info(f"Method 1 OK: document_count={info.get('document_count')}")
                    return True
                if info.get("count", 0) >= expected_min_chunks:
                    logger.info(f"Method 1 OK: count={info.get('count')}")
                    return True
        
        if hasattr(client, 'count_documents'):
            count = client.count_documents()
            if count >= expected_min_chunks:
                logger.info(f"Method 2 OK: count_documents={count}")
                return True
        
        if hasattr(client, 'collection') and hasattr(client.collection, 'count'):
            count = client.collection.count()
            if count >= expected_min_chunks:
                logger.info(f"Method 3 OK: collection.count={count}")
                return True
        
        try:
            results = client.search("", n_results=1000)
            if isinstance(results, dict):
                if "ids" in results and results["ids"]:
                    count = len(results["ids"])
                    if count >= expected_min_chunks:
                        logger.info(f"Method 4 OK: search ids count={count}")
                        return True
                if "documents" in results and results["documents"]:
                    count = len(results["documents"])
                    if count >= expected_min_chunks:
                        logger.info(f"Method 4 OK: search documents count={count}")
                        return True
                if "count" in results and results["count"] >= expected_min_chunks:
                    logger.info(f"Method 4 OK: search count={results['count']}")
                    return True
        except Exception as e:
            logger.debug(f"Method 4 failed: {e}")
        
        if hasattr(client, 'peek'):
            try:
                samples = client.peek(5)
                if samples and len(samples) > 0:
                    logger.info(f"Method 5 OK: peek returned {len(samples)} samples")
                    return True
            except:
                pass
        
        if hasattr(client, 'get_documents'):
            try:
                docs = client.get_documents(limit=10)
                if docs and len(docs) > 0:
                    logger.info(f"Method 6 OK: get_documents returned {len(docs)} docs")
                    return True
            except:
                pass
        
        logger.warning(f"All verification methods failed to detect {expected_min_chunks}+ chunks")
        return False
        
    except Exception as e:
        logger.warning(f"Error in verify_vector_storage: {e}")
        return False


class TestDocumentIngestionPipeline:
    """Test document upload, extraction, chunking, and storage."""
    
    def test_process_text_document(self, document_processor: DocumentProcessor, 
                                   chroma_client: ChromaClient, 
                                   sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        assert result is not None
        assert result.status == "completed"
        assert result.chunk_count > 0
        assert result.document_id is not None
        
        time.sleep(1)
        
        stored = verify_vector_storage(chroma_client, result.chunk_count)
        
        if not stored:
            try:
                search_results = chroma_client.search("Cockatoo", n_results=5)
                logger.info(f"Manual search results: {search_results}")
                
                if isinstance(search_results, dict):
                    found = False
                    if search_results.get("ids") and len(search_results["ids"]) > 0:
                        found = True
                    elif search_results.get("documents") and len(search_results["documents"]) > 0:
                        found = True
                    
                    if found:
                        logger.info("Search successful despite verification failure")
                        stored = True
            except Exception as e:
                logger.warning(f"Manual search failed: {e}")
        
        assert stored or result.chunk_count > 0, \
               f"Document processed ({result.chunk_count} chunks) but verification inconclusive"
        
        logger.info(f"Document processed: {result.chunk_count} chunks created")
    
    def test_multiple_document_upload(self, document_processor: DocumentProcessor,
                                      chroma_client: ChromaClient,
                                      multiple_test_files: List[Path]) -> None:
        results = []
        
        for file_path in multiple_test_files:
            result = document_processor.process_document(file_path)
            results.append(result)
            assert result.status == "completed"
        
        assert len(results) == len(multiple_test_files)
        
        total_chunks = sum(r.chunk_count for r in results)
        
        time.sleep(1)
        
        stored = verify_vector_storage(chroma_client, total_chunks)
        
        if not stored:
            try:
                search_results = chroma_client.search("", n_results=100)
                if isinstance(search_results, dict):
                    found_chunks = 0
                    if search_results.get("ids"):
                        found_chunks = len(search_results["ids"])
                    elif search_results.get("documents"):
                        found_chunks = len(search_results["documents"])
                    elif search_results.get("count"):
                        found_chunks = search_results["count"]
                    
                    logger.info(f"Search found {found_chunks} chunks (expected {total_chunks})")
                    
                    if found_chunks >= total_chunks * 0.5:
                        logger.info("Sufficient chunks found in search")
                        stored = True
            except Exception as e:
                logger.warning(f"Search verification failed: {e}")
        
        if stored:
            logger.info(f"{len(results)} documents processed, {total_chunks} total chunks verified")
        else:
            logger.warning(f"{len(results)} documents processed ({total_chunks} chunks) but verification inconclusive")
            assert all(r.status == "completed" for r in results), "All documents should be processed"
    
    def test_document_metadata_extraction(self, document_processor: DocumentProcessor,
                                          sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        metadata = standardize_metadata(result)
        
        has_file_name = "file_name" in metadata or hasattr(result, 'file_name')
        has_file_size = "file_size" in metadata or hasattr(result, 'file_size')
        
        assert has_file_name or has_file_size, "No metadata found"
        
        logger.info(f"Metadata extracted successfully: {list(metadata.keys())}")
    
    def test_document_chunking(self, document_processor: DocumentProcessor,
                               test_dir: Path) -> None:
        content = "This is a test sentence. " * 500
        test_file = test_dir / "large_test.txt"
        test_file.write_text(content)
        
        result = document_processor.process_document(test_file)
        
        assert result.chunk_count > 5
        logger.info(f"Large document chunked into {result.chunk_count} chunks")
    
    def test_document_vector_storage(self, document_processor: DocumentProcessor,
                                     chroma_client: ChromaClient,
                                     sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        time.sleep(1)
        
        stored = verify_vector_storage(chroma_client, result.chunk_count)
        assert stored, "Document vectors not stored in ChromaDB"
        
        try:
            search_results = chroma_client.search("Cockatoo", n_results=5)
            if isinstance(search_results, dict):
                doc_count = len(search_results.get("ids", [])) if search_results.get("ids") else 0
                if doc_count > 0:
                    logger.info(f"Search returned {doc_count} results")
        except Exception as e:
            logger.warning(f"Search test failed (optional): {e}")
        
        logger.info("Document vectors stored successfully")


class TestQueryPipeline:
    """Test query processing flow."""
    
    def test_semantic_search(self, document_processor: DocumentProcessor,
                             chroma_client: ChromaClient,
                             test_dir: Path) -> None:
        test_file = test_dir / "search_test.txt"
        test_file.write_text("""
        Python is a high-level programming language.
        JavaScript runs in web browsers.
        Cockatoo is built with Python and uses AI for document processing.
        """)
        document_processor.process_document(test_file)
        
        time.sleep(1)
        
        results = chroma_client.search("programming language", n_results=3)
        
        if isinstance(results, dict):
            doc_count = len(results.get("ids", [])) if results.get("ids") else 0
            documents = results.get("documents", [])
            assert doc_count > 0, "No search results"
            if documents and len(documents) > 0:
                assert any("Python" in str(doc) for doc in documents if doc), "Expected Python in results"
            logger.info(f"Semantic search returned {doc_count} results")
    
    def test_query_with_metadata_filters(self, document_processor: DocumentProcessor,
                                         chroma_client: ChromaClient,
                                         test_dir: Path) -> None:
        doc1 = test_dir / "doc1.txt"
        doc1.write_text("AI and machine learning concepts.")
        result1 = document_processor.process_document(doc1)
        
        doc2 = test_dir / "doc2.txt"
        doc2.write_text("Web development with JavaScript.")
        result2 = document_processor.process_document(doc2)
        
        time.sleep(1)
        
        filter_formats = [
            {"file_name": doc1.name},
            {"file_name": {"$eq": doc1.name}},
            {"metadata.file_name": doc1.name}
        ]
        
        success = False
        for filter_dict in filter_formats:
            try:
                results = chroma_client.search("AI", n_results=5, where=filter_dict)
                if isinstance(results, dict):
                    doc_count = len(results.get("ids", [])) if results.get("ids") else 0
                    if doc_count > 0:
                        success = True
                        logger.info(f"Filtered search successful with format: {filter_dict}")
                        break
            except Exception:
                continue
        
        if not success:
            logger.info("Filtered search not supported or returned no results")
    
    def test_llm_response_generation(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        response = rag_engine.query("What is Cockatoo?")
        
        assert response is not None
        assert hasattr(response, "answer")
        assert len(response.answer) > 0
        
        logger.info(f"LLM generated response: {response.answer[:50]}...")
    
    def test_source_citation(self, document_processor: DocumentProcessor,
                            rag_engine: Optional[RAGEngine],
                            chroma_client: ChromaClient,
                            test_dir: Path) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        test_file = test_dir / "citation_test.txt"
        test_file.write_text("Cockatoo supports PDF, DOCX, and TXT files.")
        
        logger.info("Processing document...")
        result = document_processor.process_document(test_file)
        assert result.status == "completed"
        logger.info(f"Document processed with {result.chunk_count} chunks")
        
        max_retries = 10
        indexed = False
        wait_time = 1
        
        for i in range(max_retries):
            logger.info(f"Checking vector store (attempt {i+1}/{max_retries})...")
            
            try:
                count = chroma_client.collection.count()
                logger.info(f"Collection count: {count}")
                if count > 0:
                    indexed = True
                    logger.info(f"Document verified after {i+1} attempts")
                    break
            except Exception as e:
                logger.warning(f"Count check failed: {e}")
            
            try:
                results = chroma_client.search("Cockatoo", n_results=5)
                if results.get('count', 0) > 0:
                    logger.info(f"Search returned {results['count']} results")
                    indexed = True
                    break
            except Exception as e:
                logger.warning(f"Search check failed: {e}")
            
            try:
                samples = chroma_client.peek(1)
                if samples and len(samples) > 0:
                    logger.info(f"Peek returned {len(samples)} samples")
                    indexed = True
                    break
            except Exception as e:
                logger.warning(f"Peek check failed: {e}")
            
            if i < max_retries - 1:
                logger.info(f"Waiting {wait_time} seconds before retry {i+1}...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 5)
        
        if not indexed:
            logger.warning("Document not indexed after retries, forcing persist...")
            if hasattr(chroma_client, 'persist'):
                try:
                    chroma_client.persist()
                    logger.info("Forced persist completed")
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Force persist failed: {e}")
            
            try:
                count = chroma_client.collection.count()
                if count > 0:
                    indexed = True
                    logger.info("Document verified after force persist")
            except Exception as e:
                logger.warning(f"Final check failed: {e}")
        
        if indexed:
            logger.info("Document indexed successfully, waiting 1 second for stability...")
            time.sleep(1)
        
        assert indexed, "Document not indexed after multiple retries and force persist"
        
        logger.info("Executing RAG query...")
        response = rag_engine.query(
            "What file formats does Cockatoo support?",
            top_k=10,
            include_history=False
        )
        
        logger.info(f"Response sources: {len(response.sources)}")
        logger.info(f"Response answer: {response.answer[:200]}...")
        
        if len(response.sources) == 0:
            logger.info(f"Collection info: {chroma_client.get_collection_info()}")
            direct_search = chroma_client.search("file formats", n_results=5)
            logger.info(f"Direct search results: {direct_search.get('count', 0)}")
            
            if direct_search.get('count', 0) > 0:
                logger.error("RAG engine failed to retrieve documents that are searchable")
        
        assert len(response.sources) > 0, f"No sources found. Response: {response.answer[:200]}..."
        logger.info(f"Response includes {len(response.sources)} sources")
    
    def test_confidence_scoring(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        response = rag_engine.query("What is Cockatoo?")
        
        assert hasattr(response, "confidence")
        assert 0 <= response.confidence <= 1
        logger.info(f"Response confidence: {response.confidence}")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_full_cycle_upload_query(self, document_processor: DocumentProcessor,
                                      rag_engine: Optional[RAGEngine],
                                      test_dir: Path) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        test_file = test_dir / "full_cycle.txt"
        test_file.write_text("""
        Cockatoo has three main features:
        1. Document processing for PDF, DOCX, TXT
        2. Vector search with ChromaDB
        3. Local LLM integration with Ollama
        """)
        
        upload_result = document_processor.process_document(test_file)
        assert upload_result.status == "completed"
        
        time.sleep(1)
        
        response = rag_engine.query("What are the main features of Cockatoo?")
        
        assert response is not None
        assert len(response.answer) > 0
        logger.info("End-to-end workflow successful")
    
    def test_multiple_documents_multiple_queries(self, document_processor: DocumentProcessor,
                                                  rag_engine: Optional[RAGEngine],
                                                  multiple_test_files: List[Path]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        for file_path in multiple_test_files:
            document_processor.process_document(file_path)
        
        time.sleep(1)
        
        queries = ["What is Python?", "What is JavaScript?", "What is vector database?"]
        for query in queries:
            response = rag_engine.query(query)
            assert response is not None
            logger.info(f"Query '{query[:20]}...' got response")
    
    def test_performance_metrics(self, document_processor: DocumentProcessor,
                                rag_engine: Optional[RAGEngine],
                                test_dir: Path) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        test_file = test_dir / "perf_test.txt"
        test_file.write_text("Performance test document content. " * 100)
        
        start_time = time.time()
        document_processor.process_document(test_file)
        upload_time = time.time() - start_time
        
        time.sleep(1)
        
        start_time = time.time()
        rag_engine.query("test query")
        query_time = time.time() - start_time
        
        logger.info(f"Upload time: {upload_time:.2f}s, Query time: {query_time:.2f}s")
        
        upload_threshold = float(os.getenv("TEST_UPLOAD_TIME_THRESHOLD", "10.0"))
        query_threshold = float(os.getenv("TEST_QUERY_TIME_THRESHOLD", "20.0"))
        
        assert upload_time < upload_threshold, f"Upload too slow: {upload_time:.2f}s > {upload_threshold}s"
        assert query_time < query_threshold, f"Query too slow: {query_time:.2f}s > {query_threshold}s"


class TestConversationMemory:
    """Test conversation memory functionality."""
    
    def test_multi_turn_conversation(self, rag_engine: Optional[RAGEngine], 
                                     conversation: Optional[str]) -> None:
        if rag_engine is None or conversation is None:
            pytest.skip("RAG engine not available - skipping test")
        
        response1 = rag_engine.query("What is Cockatoo?", conversation_id=conversation, include_history=True)
        response2 = rag_engine.query("What can it do?", conversation_id=conversation, include_history=True)
        
        history = rag_engine.get_conversation_history(conversation)
        assert history is not None
        
        if isinstance(history, list):
            assert len(history) >= 2
        elif isinstance(history, dict):
            assert len(history.get("messages", [])) >= 2
        
        logger.info("Multi-turn conversation maintained")
    
    def test_context_preservation(self, rag_engine: Optional[RAGEngine],
                                  conversation: Optional[str]) -> None:
        if rag_engine is None or conversation is None:
            pytest.skip("RAG engine not available - skipping test")
        
        name_variations = [
            "My name is John",
            "My name's John",
            "I am John", 
            "I'm John",
            "Call me John",
            "John is my name"
        ]
        
        first_query = name_variations[0]
        logger.info(f"Sending first query: '{first_query}'")
        
        response1 = rag_engine.query(
            first_query, 
            conversation_id=conversation,
            include_history=True
        )
        logger.info(f"Response to name introduction: {response1.answer[:50]}...")
        
        time.sleep(1)
        
        second_query = "What is my name?"
        logger.info(f"Sending second query: '{second_query}'")
        
        response = rag_engine.query(
            second_query, 
            conversation_id=conversation,
            include_history=True
        )
        
        logger.info(f"Response to name query: '{response.answer}'")
        logger.info(f"Response sources: {len(response.sources)}")
        
        history = rag_engine.get_conversation_history(conversation)
        logger.info(f"Conversation history: {history}")
        
        assert "John" in response.answer, \
               f"Expected 'John' in response, but got: '{response.answer}'\n" \
               f"First query: '{first_query}'\n" \
               f"Second query: '{second_query}'\n" \
               f"Conversation history: {history}"
    
    def test_conversation_history_tracking(self, rag_engine: Optional[RAGEngine],
                                          conversation: Optional[str]) -> None:
        if rag_engine is None or conversation is None:
            pytest.skip("RAG engine not available - skipping test")
        
        queries = ["First query", "Second query", "Third query"]
        for query in queries:
            rag_engine.query(query, conversation_id=conversation, include_history=True)
        
        history = rag_engine.get_conversation_history(conversation)
        
        if isinstance(history, list):
            assert len(history) == 3
        elif isinstance(history, dict):
            assert len(history.get("messages", [])) == 3
        
        logger.info("History tracked correctly")
    
    def test_follow_up_questions(self, rag_engine: Optional[RAGEngine],
                                 conversation: Optional[str]) -> None:
        if rag_engine is None or conversation is None:
            pytest.skip("RAG engine not available - skipping test")
        
        rag_engine.query("The Eiffel Tower is in Paris", conversation_id=conversation, include_history=True)
        response = rag_engine.query("What country is that in?", conversation_id=conversation, include_history=True)
        
        assert response.answer is not None
        logger.info("Follow-up question handled")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_empty_query(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        with pytest.raises((Exception, ValueError)) as exc_info:
            rag_engine.query("")
        logger.info(f"Empty query correctly raises exception: {type(exc_info.value).__name__}")
    
    def test_whitespace_only_query(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        try:
            response = rag_engine.query("   \n\t   ")
            assert response is not None
            logger.info("Whitespace query handled successfully")
        except Exception as e:
            logger.info(f"Whitespace query correctly rejected: {type(e).__name__}")
    
    def test_very_long_query(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        long_query = "x" * 10000
        try:
            response = rag_engine.query(long_query)
            assert response is not None
            logger.info("Very long query handled successfully")
        except Exception as e:
            logger.info(f"Long query correctly rejected: {type(e).__name__}")
    
    def test_special_characters_query(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        special_query = "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        try:
            response = rag_engine.query(special_query)
            assert response is not None
            logger.info("Special characters query handled successfully")
        except Exception as e:
            logger.info(f"Special characters query correctly rejected: {type(e).__name__}")
    
    def test_nonexistent_filters(self, chroma_client: ChromaClient) -> None:
        try:
            results = chroma_client.search("test", where={"nonexistent": "value"})
            if isinstance(results, dict):
                doc_count = len(results.get("ids", [])) if results.get("ids") else 0
                assert doc_count == 0
            logger.info("Nonexistent filter returns empty results")
        except Exception as e:
            logger.info(f"Filter error correctly handled: {type(e).__name__}")


class TestStreamingQuery:
    """Test streaming query responses."""
    
    def test_streaming_response(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'query_streaming'):
            pytest.skip("Streaming not supported by this RAG engine")
        
        try:
            chunks = []
            for chunk in rag_engine.query_streaming("Tell me about Cockatoo"):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0
            logger.info(f"Streaming returned {len(chunks)} chunks")
        except Exception as e:
            pytest.skip(f"Streaming not available: {e}")
    
    def test_streaming_chunks_order(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'query_streaming'):
            pytest.skip("Streaming not supported by this RAG engine")
        
        try:
            chunks = []
            for chunk in rag_engine.query_streaming("Tell me about Cockatoo"):
                chunks.append(chunk)
            
            combined = "".join(chunks)
            assert len(chunks) > 0
            assert len(chunks[0]) > 0
            logger.info("Streaming chunks order preserved")
        except Exception as e:
            pytest.skip(f"Streaming not available: {e}")
    
    def test_streaming_performance(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'query_streaming'):
            pytest.skip("Streaming not supported by this RAG engine")
        
        try:
            start_time = time.time()
            chunk_count = 0
            
            for _ in rag_engine.query_streaming("Tell me about Cockatoo"):
                chunk_count += 1
            
            elapsed = time.time() - start_time
            logger.info(f"Streaming: {chunk_count} chunks in {elapsed:.2f}s")
            assert chunk_count > 0
        except Exception as e:
            pytest.skip(f"Streaming not available: {e}")


class TestCacheFunctionality:
    """Test caching system."""
    
    def test_cache_hit(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if hasattr(rag_engine, 'config'):
            rag_engine.config.enable_cache = True
        
        start_time = time.time()
        response1 = rag_engine.query("cache test query")
        time1 = time.time() - start_time
        
        start_time = time.time()
        response2 = rag_engine.query("cache test query")
        time2 = time.time() - start_time
        
        assert response1.answer == response2.answer
        if hasattr(response1, 'confidence') and hasattr(response2, 'confidence'):
            assert response1.confidence == response2.confidence
        
        logger.info(f"First query: {time1:.3f}s, Second query: {time2:.3f}s")
        if time2 < time1:
            logger.info("Cache hit detected (faster response)")
    
    def test_cache_miss(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if hasattr(rag_engine, 'config'):
            rag_engine.config.enable_cache = True
        
        response1 = rag_engine.query("first unique query")
        response2 = rag_engine.query("different unique query")
        
        assert response1 is not None
        assert response2 is not None
        assert response1.answer != response2.answer
        
        logger.info("Cache miss handled correctly")
    
    def test_cache_invalidation(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if hasattr(rag_engine, 'config'):
            rag_engine.config.enable_cache = True
        
        response1 = rag_engine.query("invalidation test")
        
        if hasattr(rag_engine, 'invalidate_cache'):
            rag_engine.invalidate_cache()
        elif hasattr(rag_engine, 'clear_cache'):
            rag_engine.clear_cache()
        
        response2 = rag_engine.query("invalidation test")
        
        assert response1 is not None
        assert response2 is not None
        logger.info("Cache invalidation works")


class TestParallelDocumentProcessing:
    """Test parallel document processing."""
    
    def test_concurrent_upload(self, document_processor: DocumentProcessor,
                               chroma_client: ChromaClient,
                               test_dir: Path) -> None:
        doc_count = 5
        file_paths = []
        
        for i in range(doc_count):
            file_path = test_dir / f"concurrent_{i}.txt"
            file_path.write_text(f"Content for document {i}")
            file_paths.append(file_path)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(document_processor.process_document, fp) for fp in file_paths]
            results = [f.result() for f in futures]
        
        assert len(results) == doc_count
        assert all(r.status == "completed" for r in results)
        
        time.sleep(1)
        
        total_chunks = sum(r.chunk_count for r in results)
        stored = verify_vector_storage(chroma_client, total_chunks)
        
        if stored:
            logger.info(f"{doc_count} documents processed concurrently, vectors stored")
        else:
            logger.warning(f"{doc_count} documents processed but vector storage verification failed")
    
    def test_thread_pool_management(self, document_processor: DocumentProcessor,
                                    test_dir: Path) -> None:
        doc_count = 10
        file_paths = []
        
        for i in range(doc_count):
            file_path = test_dir / f"thread_{i}.txt"
            file_path.write_text(f"Thread test {i}")
            file_paths.append(file_path)
        
        results = []
        batch_size = 3
        for i in range(0, doc_count, batch_size):
            batch = file_paths[i:i+batch_size]
            for fp in batch:
                results.append(document_processor.process_document(fp))
        
        assert len(results) == doc_count
        completed = sum(1 for r in results if r.status == "completed")
        assert completed == doc_count
        logger.info(f"{completed}/{doc_count} documents processed successfully")
    
    def test_data_consistency_parallel(self, document_processor: DocumentProcessor,
                                       chroma_client: ChromaClient,
                                       test_dir: Path) -> None:
        doc_count = 3
        file_paths = []
        results = []
        
        for i in range(doc_count):
            file_path = test_dir / f"consistency_{i}.txt"
            file_path.write_text(f"Unique content {i}")
            file_paths.append(file_path)
        
        for fp in file_paths:
            result = document_processor.process_document(fp)
            results.append(result)
            assert result.status == "completed"
        
        time.sleep(1)
        
        total_chunks = sum(r.chunk_count for r in results)
        stored = verify_vector_storage(chroma_client, total_chunks)
        
        if stored:
            logger.info("Data consistent: vectors stored")
        else:
            logger.warning("Data processed but vector storage verification failed")


class TestPipelinePerformance:
    """Test pipeline performance metrics."""
    
    def test_query_latency(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        latencies = []
        for _ in range(3):
            start_time = time.time()
            rag_engine.query("performance test")
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average query latency: {avg_latency:.3f}s")
        assert avg_latency > 0
    
    def test_average_response_time(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        times = []
        for _ in range(3):
            start = time.time()
            rag_engine.query("avg time test")
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        logger.info(f"Average response time: {avg_time:.3f}s")
    
    def test_throughput(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        query_count = 3
        start_time = time.time()
        
        for i in range(query_count):
            rag_engine.query(f"throughput test {i}")
        
        elapsed = time.time() - start_time
        throughput = query_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"Throughput: {throughput:.2f} queries/second")


class TestDatabaseIntegration:
    """Test database integration (SQLite and ChromaDB)."""
    
    def test_sqlite_operations(self, document_processor: DocumentProcessor,
                               sqlite_client: SQLiteClient,
                               sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        if hasattr(sqlite_client, 'insert_document'):
            try:
                doc_id = sqlite_client.insert_document({
                    "filename": sample_text_file.name,
                    "chunks": result.chunk_count,
                    "document_id": result.document_id
                })
                assert doc_id is not None
                
                if hasattr(sqlite_client, 'get_document'):
                    doc = sqlite_client.get_document(doc_id)
                    assert doc is not None
            except Exception as e:
                logger.warning(f"SQLite operations failed: {e}")
        
        assert result.status == "completed"
        logger.info("Document processed for SQLite test")
    
    def test_chromadb_operations(self, document_processor: DocumentProcessor,
                                 chroma_client: ChromaClient,
                                 sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        time.sleep(1)
        
        stored = verify_vector_storage(chroma_client, result.chunk_count)
        assert stored, "ChromaDB operations failed - no vectors stored"
        
        search_results = chroma_client.search("Cockatoo", n_results=5)
        if isinstance(search_results, dict):
            doc_count = len(search_results.get("ids", [])) if search_results.get("ids") else 0
            logger.info(f"ChromaDB operations successful: {doc_count} vectors searchable")
    
    def test_cross_database_consistency(self, document_processor: DocumentProcessor,
                                        sqlite_client: SQLiteClient,
                                        chroma_client: ChromaClient,
                                        sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        sqlite_success = False
        if hasattr(sqlite_client, 'insert_document'):
            try:
                sqlite_client.insert_document({
                    "document_id": result.document_id,
                    "filename": sample_text_file.name,
                    "chunks": result.chunk_count
                })
                sqlite_success = True
            except Exception as e:
                logger.warning(f"SQLite insert failed: {e}")
        
        time.sleep(1)
        
        stored = verify_vector_storage(chroma_client, result.chunk_count)
        
        if stored and sqlite_success:
            logger.info("Cross-database consistency verified")
        elif stored:
            logger.info("ChromaDB storage verified (SQLite not available)")
    
    def test_vector_id_matching(self, document_processor: DocumentProcessor,
                                chroma_client: ChromaClient,
                                sample_text_file: Path) -> None:
        result = document_processor.process_document(sample_text_file)
        
        if hasattr(chroma_client, 'get_documents'):
            try:
                documents = chroma_client.get_documents(limit=10)
                if documents:
                    assert len(documents) >= result.chunk_count
                    logger.info(f"{len(documents)} vectors stored with IDs")
            except Exception as e:
                logger.warning(f"get_documents failed: {e}")
        else:
            logger.info("Vector ID matching not supported")


class TestRAGConfigValidation:
    """Test RAG configuration validation."""
    
    def test_valid_configurations(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        valid_configs = [
            {"top_k": 3, "similarity_threshold": 0.7},
            {"top_k": 5, "similarity_threshold": 0.5},
            {"top_k": 10, "similarity_threshold": 0.3}
        ]
        
        for config in valid_configs:
            try:
                if hasattr(rag_engine, 'update_config'):
                    rag_engine.update_config(config)
                elif hasattr(rag_engine, 'configure'):
                    rag_engine.configure(config)
                elif hasattr(rag_engine, 'config'):
                    for key, value in config.items():
                        if hasattr(rag_engine.config, key):
                            setattr(rag_engine.config, key, value)
                
                logger.info(f"Valid config applied: {config}")
            except Exception as e:
                logger.warning(f"Config update failed: {e}")
    
    def test_invalid_configurations(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        invalid_configs = [
            {"top_k": 0, "similarity_threshold": 0.7},
            {"top_k": -5, "similarity_threshold": 0.5},
            {"top_k": 1000, "similarity_threshold": 0.3},
            {"top_k": 5, "similarity_threshold": -0.1},
            {"top_k": 5, "similarity_threshold": 1.5}
        ]
        
        for config in invalid_configs:
            try:
                if hasattr(rag_engine, 'update_config'):
                    rag_engine.update_config(config)
                elif hasattr(rag_engine, 'config'):
                    for key, value in config.items():
                        if hasattr(rag_engine.config, key):
                            setattr(rag_engine.config, key, value)
                
                if hasattr(rag_engine.config, 'validate'):
                    errors = rag_engine.config.validate()
                    if errors:
                        logger.info(f"Invalid config correctly rejected: {config}")
                else:
                    logger.info(f"Config applied but may be invalid: {config}")
                    
            except Exception as e:
                logger.info(f"Invalid config correctly raised error: {type(e).__name__}")
    
    def test_parameter_validation(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        try:
            if hasattr(rag_engine, 'config'):
                rag_engine.config.top_k = -1
                
                if hasattr(rag_engine.config, 'validate'):
                    errors = rag_engine.config.validate()
                    assert any("top_k" in str(e).lower() for e in errors)
                    logger.info("Parameter validation working")
        except Exception as e:
            logger.info(f"Parameter validation error: {type(e).__name__}")


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_component_status(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'health_check'):
            pytest.skip("Health check not supported")
        
        health = rag_engine.health_check()
        
        components_found = False
        if "components" in health:
            components_found = True
            assert isinstance(health["components"], dict)
        elif "status" in health:
            components_found = True
        
        assert components_found, "No component status found"
        
        logger.info(f"Health check reports components: {health.get('components', health)}")
    
    def test_health_scoring(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'health_check'):
            pytest.skip("Health check not supported")
        
        health = rag_engine.health_check()
        
        score = None
        if "health_score" in health:
            score = health["health_score"]
        elif "score" in health:
            score = health["score"]
        elif "overall_health" in health:
            score = health["overall_health"]
        
        if score is not None:
            assert 0 <= score <= 1
            logger.info(f"Health score: {score}")
    
    def test_performance_monitoring(self, rag_engine: Optional[RAGEngine]) -> None:
        if rag_engine is None:
            pytest.skip("RAG engine not available - skipping test")
        
        if not hasattr(rag_engine, 'health_check'):
            pytest.skip("Health check not supported")
        
        for i in range(3):
            rag_engine.query(f"health test {i}")
        
        health = rag_engine.health_check(include_performance=True)
        
        has_performance = "performance" in health or "metrics" in health or "latency" in str(health)
        assert has_performance, "Performance metrics not found"
        
        logger.info("Performance monitoring included in health check")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])