import base64
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from app.config import MAX_MEDIA_MB
from app.database import db
from app.brain.doc_brain import doc_brain
from app.brain.safety import get_workspace_dir


INGEST_DIR = Path(get_workspace_dir()) / "backend" / "local_data" / "ingested_files"


def _safe_filename(name: str, fallback_ext: str = ".txt") -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._")
    if not clean:
        clean = f"document_{uuid.uuid4().hex[:8]}{fallback_ext}"
    return clean[:140]


def _default_title_from_source(source: str) -> str:
    raw = Path(source).name or source
    raw = raw.split("?")[0].split("#")[0]
    title = re.sub(r"[_-]+", " ", raw).strip()
    return title or f"Document {uuid.uuid4().hex[:8]}"


def _guess_source_type(path: Path, mime_type: Optional[str] = None, requested: str = "pdf") -> str:
    ext = path.suffix.lower()
    mime = (mime_type or "").lower()
    if ext == ".pdf" or "pdf" in mime:
        return "pdf"
    if ext == ".docx" or "wordprocessingml" in mime:
        return "pdf"
    if ext in {".txt", ".md", ".csv", ".json"} or mime.startswith("text/"):
        return requested or "url"
    return requested or "url"


def _ensure_size_ok(raw_size_bytes: int) -> Optional[str]:
    max_bytes = MAX_MEDIA_MB * 1024 * 1024
    if raw_size_bytes > max_bytes:
        return f"File is too large ({raw_size_bytes / 1024 / 1024:.1f} MB). Limit is {MAX_MEDIA_MB} MB."
    return None


def extract_text_from_file(file_path: str, mime_type: Optional[str] = None) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    size_error = _ensure_size_ok(path.stat().st_size)
    if size_error:
        raise ValueError(size_error)

    ext = path.suffix.lower()
    mime = (mime_type or "").lower()
    if ext == ".pdf" or "pdf" in mime:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("PDF ingestion needs `pypdf`. Install backend requirements first.") from e
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {idx + 1}]\n{text.strip()}")
        return "\n\n".join(pages).strip()

    if ext == ".docx" or "wordprocessingml" in mime:
        try:
            import docx
        except ImportError as e:
            raise RuntimeError("DOCX ingestion needs `python-docx`. Install backend requirements first.") from e
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()

    return path.read_text(encoding="utf-8", errors="ignore").strip()


def ingest_text_document(
    title: str,
    content: str,
    source_type: str = "url",
    source_url: Optional[str] = None,
) -> str:
    title = (title or "").strip() or f"Document {uuid.uuid4().hex[:8]}"
    content = (content or "").strip()
    if not content:
        return "Could not ingest document: extracted text is empty."

    result = doc_brain.ingest_document(
        title=title,
        content=content,
        source_type=source_type or "url",
        source_url=source_url
    )
    chunks = db.get_document_chunks(title)
    return (
        f"Ingested '{result['title']}' into vector memory.\n"
        f"Document ID: {result['doc_id']}\n"
        f"Chunks saved: {result['chunks_count']}\n"
        f"SQLite verification chunks: {len(chunks)}\n"
        f"Source: {result['source']}"
    )


def ingest_local_file(
    file_path: str,
    title: Optional[str] = None,
    source_type: str = "pdf",
    source_url: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> str:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path(get_workspace_dir()) / path
    path = path.resolve()

    text = extract_text_from_file(str(path), mime_type=mime_type)
    resolved_title = title or _default_title_from_source(str(path))
    resolved_source_type = _guess_source_type(path, mime_type=mime_type, requested=source_type)
    return ingest_text_document(
        title=resolved_title,
        content=text,
        source_type=resolved_source_type,
        source_url=source_url or str(path)
    )


def download_url_to_ingest_file(url: str, filename: Optional[str] = None) -> Tuple[Path, str]:
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    extension = ".pdf" if "pdf" in content_type.lower() else Path(url.split("?")[0]).suffix or ".txt"
    target_name = _safe_filename(filename or Path(url.split("?")[0]).name, extension)
    if "." not in target_name:
        target_name += extension
    target = INGEST_DIR / f"{uuid.uuid4().hex[:8]}_{target_name}"

    max_bytes = MAX_MEDIA_MB * 1024 * 1024
    total = 0
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                try:
                    target.unlink()
                except Exception:
                    pass
                raise ValueError(f"Download exceeded {MAX_MEDIA_MB} MB limit.")
            f.write(chunk)
    return target, content_type


def ingest_url_document(
    url: str,
    title: Optional[str] = None,
    source_type: str = "pdf",
) -> str:
    target, content_type = download_url_to_ingest_file(url)
    resolved_title = title or _default_title_from_source(url)
    return ingest_local_file(
        file_path=str(target),
        title=resolved_title,
        source_type=source_type,
        source_url=url,
        mime_type=content_type,
    )


def ingest_media_document(
    media_base64: str,
    filename: Optional[str] = None,
    title: Optional[str] = None,
    mime_type: Optional[str] = None,
    source_type: str = "whatsapp",
) -> str:
    if not media_base64:
        return "Could not ingest WhatsApp document: no media data was received."
    try:
        raw = base64.b64decode(media_base64)
        size_error = _ensure_size_ok(len(raw))
        if size_error:
            return size_error

        INGEST_DIR.mkdir(parents=True, exist_ok=True)
        extension = ".pdf" if "pdf" in (mime_type or "").lower() else Path(filename or "").suffix or ".txt"
        safe = _safe_filename(filename or f"whatsapp_document{extension}", extension)
        target = INGEST_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
        target.write_bytes(raw)

        resolved_title = title or _default_title_from_source(filename or str(target))
        return ingest_local_file(
            file_path=str(target),
            title=resolved_title,
            source_type=source_type,
            source_url=f"whatsapp://{safe}",
            mime_type=mime_type,
        )
    except Exception as e:
        return f"Could not ingest WhatsApp document: {e}"


def list_ingested_documents() -> str:
    docs = db.get_all_documents()
    if not docs:
        return "No documents are currently ingested."
    lines = ["Ingested documents:"]
    for doc in docs[:20]:
        lines.append(
            f"- {doc['title']} | {doc['source_type']} | {doc['chunk_count']} chunks | {doc['source']}"
        )
    return "\n".join(lines)
