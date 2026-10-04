"""Document upload and retrieval endpoints."""

import asyncio
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.api.events import publish
from backend.api.schemas import (
    ChunkDeletedOut,
    ChunksClearedOut,
    DocumentListOut,
    DocumentQueryIn,
    DocumentQueryOut,
    LinkOut,
    UploadOut,
)
from backend.core import docstore, rag, storage

router = APIRouter(tags=["Documents"])


@router.post("/documents/upload", response_model=UploadOut)
async def upload_document(
    file: UploadFile = File(...),
    caller_id: str = Form("default"),
):
    """Extract, chunk, embed and index a document, keeping the original."""
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty document upload")

    filename = file.filename or "uploaded_document"

    # The original goes to object storage so a re-index does not need the caller
    # to upload again. Failure here must not lose the document: the text and
    # vectors are what answer questions, and they are about to be written
    # regardless.
    object_key = None
    if storage.configured():
        try:
            object_key = await storage.put(
                f"uploads/{caller_id}/{uuid.uuid4().hex[:8]}-{filename}",
                content,
                file.content_type or "application/octet-stream",
            )
        except Exception as exc:
            publish({"kind": "storage_error", "filename": filename, "detail": str(exc)[:200]})

    # index_document does OCR and embedding over the network with synchronous
    # clients. Called directly it would block the event loop for the whole
    # upload, stalling every other request including live agent tool calls.
    result = await asyncio.to_thread(
        rag.index_document,
        filename=filename,
        content=content,
        mime_type=file.content_type or "",
        caller_id=caller_id,
    )

    chunks = result.pop("chunks", [])
    for row in chunks:
        row["object_key"] = object_key
    await docstore.add_chunks(chunks)

    publish({"kind": "document_uploaded", "filename": filename, "caller_id": caller_id})
    return {**result, "object_key": object_key, "stored_chunks": len(chunks)}


@router.get("/documents", response_model=DocumentListOut)
async def list_documents():
    """Indexed chunks, newest first."""
    chunks = await docstore.list_chunks()
    return {
        "documents": chunks,
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY", "").strip()),
        "storage_configured": storage.configured(),
        "total_chunks": len(chunks),
    }


@router.post("/documents/query", response_model=DocumentQueryOut)
async def query_documents(body: DocumentQueryIn):
    """Query the vector store, to test retrieval without placing a call."""
    query_text = body.query.strip()
    caller_id = body.caller_id or "default"
    top_k = body.top_k

    if not query_text:
        return JSONResponse({"error": "query is required"}, status_code=400)

    fallback = await docstore.search(query_text, caller_id=caller_id, top_k=top_k)
    result_text = rag.query_vector_store(
        query_text=query_text, caller_id=caller_id, top_k=top_k, fallback_rows=fallback
    )
    return {"query": query_text, "result": result_text,
            "caller_id": caller_id, "top_k": top_k}


@router.get("/documents/{doc_id}/link", response_model=LinkOut)
async def document_link(doc_id: str):
    """A time-limited URL for the original file behind a chunk."""
    rows = [c for c in await docstore.list_chunks(limit=1000) if c["id"] == doc_id]
    if not rows:
        raise HTTPException(404, f"unknown chunk {doc_id}")
    key = rows[0].get("object_key")
    if not key:
        raise HTTPException(404, "no original stored for this chunk")
    if not storage.configured():
        return JSONResponse(
            {"error": "Object storage is not configured.", "missing": storage.missing()},
            status_code=503,
        )
    return {"url": await storage.presign(key)}


@router.delete("/documents/{doc_id}", response_model=ChunkDeletedOut)
async def delete_document_chunk(doc_id: str):
    return {"deleted": doc_id, "count": await docstore.delete_chunk(doc_id)}


@router.delete("/documents", response_model=ChunksClearedOut)
async def clear_documents():
    return {"cleared": await docstore.clear()}
