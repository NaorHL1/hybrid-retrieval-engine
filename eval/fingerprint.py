"""Fingerprints that tell a stale baseline apart from a real regression."""
import hashlib
from pathlib import Path


def corpus_fingerprint(rows: list[dict]) -> dict:
    """Order-independent hash of every indexed document (id, title, text)."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        for field in ("id", "title", "text"):
            h.update(str(r.get(field, "")).encode("utf-8"))
            h.update(b"\x00")
    return {"n_docs": len(rows), "sha256": h.hexdigest()}


def file_sha256(path: Path) -> str:
    """Hash of a file's content with line endings normalized (git may convert LF/CRLF)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
