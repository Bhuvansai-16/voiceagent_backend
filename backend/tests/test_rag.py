"""Unit tests for the RAG engine: chunking, text extraction, local fallback vector store, and persona integration.
"""

from backend.core import personas, rag


def test_chunk_text_splits_long_content():
    sample = "Paragraph 1 is about AI voice agents.\n\nParagraph 2 is about RAG and Pinecone."
    chunks = rag.chunk_text(sample, chunk_size=30)
    assert len(chunks) >= 2
    assert "AI voice agents" in chunks[0]
    assert "Pinecone" in chunks[1]


def test_extract_text_from_plain_text_file():
    raw = b"Hello world! This is a test document."
    text = rag.extract_text_from_file("test.txt", raw)
    assert "Hello world!" in text


def test_index_and_query_document_local_fallback():
    content = b"The company wifi password is SuperSecretVoice2026."
    res = rag.index_document("wifi_policy.txt", content, caller_id="user1")
    assert res["status"] == "success"
    assert res["chunks_indexed"] >= 1

    # index_document returns the chunk rows rather than appending to a
    # module-level list; the caller persists them. Passing them straight back
    # in exercises the same round trip at the new seam.
    answer = rag.query_vector_store(
        "wifi password", caller_id="user1", fallback_rows=res["chunks"]
    )
    assert "SuperSecretVoice2026" in answer


def test_rag_persona_registration():
    all_personas = {p.id: p for p in personas.resolve({})}
    assert "rag" in all_personas
    rag_persona = all_personas["rag"]
    assert "search_documents" in rag_persona.granted_tools()
    assert rag_persona.label == "Document RAG"


def test_extract_text_with_nemotron_ocr_mocked(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-12345")

    class MockResponse:
        status_code = 200

        def json(self):
            return {"text": "Extracted text from server rack diagram: Port 42 is active."}

        def raise_for_status(self):
            pass

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers=None, json=None):
            assert "nvapi-test-key-12345" in headers["Authorization"]
            return MockResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", MockClient)

    img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    extracted = rag.extract_text_from_file("rack_diagram.png", img_data, mime_type="image/png")
    assert "Port 42 is active" in extracted


def test_index_image_document_with_nemotron_ocr_mocked(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-67890")

    class MockResponse:
        status_code = 200

        def json(self):
            return {"text": "NVIDIA Nemotron OCR result: Emergency IT support hotline is 1-800-555-0199."}

        def raise_for_status(self):
            pass

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers=None, json=None):
            return MockResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", MockClient)

    img_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    res = rag.index_document("support_flyer.jpeg", img_bytes, mime_type="image/jpeg", caller_id="ocr_user")
    assert res["status"] == "success"
    assert res["chunks_indexed"] >= 1

    answer = rag.query_vector_store(
        "emergency support hotline", caller_id="ocr_user", fallback_rows=res["chunks"]
    )
    assert "1-800-555-0199" in answer


def test_upload_document_endpoint_fastapi():
    from fastapi.testclient import TestClient
    from backend.api.main import app

    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("manual.txt", b"Router IP address is 192.168.1.254", "text/plain")},
        data={"caller_id": "fastapi_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "manual.txt"
