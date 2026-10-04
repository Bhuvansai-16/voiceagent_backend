"""RAG Engine: Document Ingestion, Mistral OCR, Vector Embeddings, and Pinecone Search.

Supports:
- Image OCR extraction via Mistral AI (mistral-ocr)
- Text / PDF / Markdown document chunking
- Embedding generation (OpenAI, Mistral, Google, or local fallback)
- Pinecone vector store upsert & query, with local fallback for offline/test environments
"""

import base64
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("service-desk.rag")

# The per-process list that used to live here is gone: see core/docstore.py.
# The API appended to it and the worker read its own empty copy, so an uploaded
# document was invisible to the agent unless Pinecone was configured.


def extract_text_with_nemotron_ocr(filename: str, content: bytes, mime_type: str = "") -> str | None:
    """Extract text from image using NVIDIA Nemotron OCR v2 endpoint."""
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVIDIA_NEMOTRON_OCR_API_KEY", "").strip()
    if not nvidia_key:
        return None

    ocr_url = os.getenv("NVIDIA_OCR_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2").strip()
    ext = Path(filename).suffix.lower()
    mime = mime_type or f"image/{ext.lstrip('.') if ext else 'png'}"
    if mime in ("image/jpg", "image/pjpeg"):
        mime = "image/jpeg"

    b64_content = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{mime};base64,{b64_content}"

    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": data_url,
            }
        ]
    }

    try:
        import httpx

        # Synchronous on purpose: extract_text_from_file is called from a worker
        # thread via asyncio.to_thread, never from the event loop directly. It
        # used to be called straight from an async handler, where a 30-second
        # OCR request blocked every other request in the process.
        logger.info("requesting NVIDIA Nemotron OCR v2 for image %s", filename)
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(ocr_url, headers=headers, json=payload)
            if resp.status_code in (400, 422):
                # Try OpenAI-style message payload as fallback
                alt_payload = {
                    "model": "nvidia/nemotron-ocr-v2",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all text and Markdown accurately."},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                }
                resp = client.post(ocr_url, headers=headers, json=alt_payload)

            resp.raise_for_status()
            data = resp.json()

            text_result = ""
            if isinstance(data, dict):
                if "text" in data:
                    text_result = str(data["text"])
                elif "data" in data and isinstance(data["data"], str):
                    text_result = data["data"]
                elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict) and "message" in first:
                        content_val = first["message"].get("content", "")
                        if isinstance(content_val, str):
                            text_result = content_val
                        elif isinstance(content_val, list):
                            text_result = "\n".join(
                                str(item.get("text", item)) for item in content_val if isinstance(item, dict)
                            )
                elif "predictions" in data and isinstance(data["predictions"], list):
                    text_result = "\n".join(str(p.get("text", p)) for p in data["predictions"] if isinstance(p, dict))
                elif "output" in data:
                    text_result = str(data["output"])

            if not text_result and isinstance(data, list):
                text_result = "\n".join(str(item) for item in data)

            extracted = text_result.strip()
            if extracted:
                logger.info("NVIDIA Nemotron OCR extracted %d characters from %s", len(extracted), filename)
                return extracted
    except Exception as exc:
        logger.error("NVIDIA Nemotron OCR failed for %s: %s", filename, exc)

    return None


def extract_text_from_file(filename: str, content: bytes, mime_type: str = "") -> str:
    """Extract readable text from uploaded file.
    
    Uses NVIDIA Nemotron OCR v2 (or Mistral OCR fallback) for image files (.png, .jpg, .jpeg, .webp, .bmp, image/*).
    Decodes plain text / markdown / code files directly.
    """
    ext = Path(filename).suffix.lower()
    is_image = mime_type.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    if is_image:
        # 1. NVIDIA Nemotron OCR v2
        if text := extract_text_with_nemotron_ocr(filename, content, mime_type):
            return text

        # 2. Mistral OCR fallback
        mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if mistral_key:
            try:
                from mistralai import Mistral
                client = Mistral(api_key=mistral_key)
                
                b64_content = base64.b64encode(content).decode("utf-8")
                mime = mime_type or f"image/{ext.lstrip('.')}"
                if mime == "image/jpg":
                    mime = "image/jpeg"
                data_url = f"data:{mime};base64,{b64_content}"

                logger.info("requesting Mistral OCR fallback for image %s", filename)
                response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={"type": "image_url", "image_url": data_url}
                )
                
                extracted_pages = []
                for page in getattr(response, "pages", []):
                    extracted_pages.append(getattr(page, "markdown", str(page)))
                
                extracted_text = "\n\n".join(extracted_pages).strip()
                if extracted_text:
                    logger.info("Mistral OCR extracted %d characters from %s", len(extracted_text), filename)
                    return extracted_text
            except Exception as exc:
                logger.error("Mistral OCR failed for %s: %s", filename, exc)
        else:
            logger.warning("Neither NVIDIA_API_KEY nor MISTRAL_API_KEY set; image %s cannot be OCR'd", filename)
            return f"[Image document {filename}: NVIDIA_API_KEY and MISTRAL_API_KEY missing for OCR processing]"

    # Default text decoder for text/pdf/markdown files
    try:
        text = content.decode("utf-8", errors="ignore").strip()
        # Clean up any non-printable null bytes
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text if text else f"[Document {filename}: empty or unreadable text]"
    except Exception as exc:
        logger.error("Could not read text from %s: %s", filename, exc)
        return f"[Document {filename}: text extraction failed]"


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into smaller chunks with overlap."""
    if not text:
        return []
    
    # Split by paragraphs or sentences where possible
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) <= chunk_size:
            current_chunk.append(p)
            current_len += len(p) + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Fallback to naive sliding window if text had no paragraph breaks and exceeded chunk_size
    if not chunks and len(text) > chunk_size:
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunks.append(text[i : i + chunk_size])

    return chunks if chunks else [text]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for input texts using available provider."""
    if not texts:
        return []

    # 1. Mistral Embeddings (mistral-embed -> 1024 dims)
    if mistral_key := os.getenv("MISTRAL_API_KEY", "").strip():
        try:
            from mistralai import Mistral
            client = Mistral(api_key=mistral_key)
            resp = client.embeddings.create(
                model="mistral-embed",
                inputs=texts,
            )
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning("Mistral embedding generation failed: %s", exc)

    # 2. Google Embeddings (text-embedding-004 -> 768 dims)
    if google_key := os.getenv("GOOGLE_API_KEY", "").strip():
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            embeddings = []
            for t in texts:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=t,
                )
                embeddings.append(res["embedding"])
            return embeddings
        except Exception as exc:
            logger.warning("Google embedding generation failed: %s", exc)

    # 3. NVIDIA Embeddings (llama-text-embed-v2 -> 1024 dims)
    if nvidia_key := os.getenv("NVIDIA_API_KEY", "").strip():
        try:
            # Assuming NVIDIA NIM provides OpenAI-compatible embeddings endpoint
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key
            )
            resp = client.embeddings.create(
                model="llama-text-embed-v2",
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in resp.data]
        except Exception as exc:
            logger.warning("NVIDIA embedding generation failed: %s", exc)

    # 4. Fallback deterministic pseudo-embeddings for testing without keys
    logger.info("Using pseudo-embeddings (dim=1024) for testing without cloud keys")
    results = []
    for t in texts:
        # Simple hash-based deterministic 1024-dim vector
        seed = sum(ord(c) for c in t[:100])
        vec = [((seed * (i + 1)) % 1000) / 1000.0 for i in range(1024)]
        results.append(vec)
    return results


def index_document(
    filename: str,
    content: bytes,
    mime_type: str = "",
    caller_id: str = "default",
) -> dict[str, Any]:
    """Extract, chunk, embed, and store document into Pinecone (with local fallback)."""
    extracted_text = extract_text_from_file(filename, content, mime_type)
    chunks = chunk_text(extracted_text)
    embeddings = generate_embeddings(chunks)

    doc_id = str(uuid.uuid4())[:8]
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "voiceagent-rag").strip()

    indexed_count = 0
    used_pinecone = False

    if pinecone_key and pinecone_index_name:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(pinecone_index_name)

            vectors_to_upsert = []
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
                vectors_to_upsert.append({
                    "id": f"{doc_id}-{i}",
                    "values": vec,
                    "metadata": {
                        "filename": filename,
                        "caller_id": caller_id,
                        "chunk_index": i,
                        "text": chunk[:1000],  # Pinecone metadata size limit safety
                    },
                })

            index.upsert(vectors=vectors_to_upsert)
            indexed_count = len(vectors_to_upsert)
            used_pinecone = True
            logger.info("Upserted %d vectors to Pinecone index %r", indexed_count, pinecone_index_name)
        except Exception as exc:
            logger.error("Pinecone upsert failed: %s; falling back to local doc store", exc)

    # Chunk rows for the caller to persist. index_document stays synchronous
    # and side-effect free with respect to storage; the API awaits the write.
    chunk_rows = [
        {
            "id": f"{doc_id}-{i}",
            "filename": filename,
            "caller_id": caller_id,
            "chunk_index": i,
            "text": chunk,
            "vector": vec,
        }
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings))
    ]
    if not used_pinecone:
        indexed_count = len(chunks)

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": indexed_count,
        "extracted_text_preview": extracted_text[:200],
        "used_pinecone": used_pinecone,
        "chunks": chunk_rows,
    }


def query_vector_store(query_text: str, caller_id: str = "default", top_k: int = 3,
                       fallback_rows: list[dict] | None = None) -> str:
    """Search the vector store for chunks relevant to query_text.

    `fallback_rows` are chunk rows the caller already fetched from the shared
    doc_chunks table, used when Pinecone is unconfigured or returns nothing.
    Passed in rather than read here so this module stays free of database
    imports."""
    if not query_text.strip():
        return "No query provided."

    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "voiceagent-rag").strip()
    query_vec = generate_embeddings([query_text])[0] if query_text else []

    matches: list[str] = []

    if pinecone_key and pinecone_index_name and query_vec:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(pinecone_index_name)
            
            res = index.query(
                vector=query_vec,
                top_k=top_k,
                include_metadata=True,
                filter={"caller_id": caller_id} if caller_id != "default" else None,
            )
            for hit in getattr(res, "matches", []):
                meta = getattr(hit, "metadata", {}) or {}
                if text := meta.get("text"):
                    fname = meta.get("filename", "document")
                    matches.append(f"[{fname}]: {text}")
        except Exception as exc:
            logger.error("Pinecone query failed: %s; using local doc store", exc)

    # Fallback to the shared chunk table when Pinecone is absent or empty.
    # Synchronous by necessity: this is called from the agent's tool, which is
    # already inside a coroutine, so the caller passes results in instead.
    if not matches and fallback_rows:
        for row in fallback_rows[:top_k]:
            matches.append(f"[{row['filename']}]: {row['text']}")

    if matches:
        return "\n\n".join(matches)
    return "No relevant information found in uploaded documents."
