import pytest

from app.core import chunk_text, extract_text


def test_chunk_text_splits_long_text_and_preserves_content():
    text = "A sentence. " * 200
    chunks = chunk_text(text, size=200, overlap=30)
    assert len(chunks) > 1
    assert all(chunks)
    assert chunks[0][:20] == text[:20]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


def test_extract_text_txt():
    assert extract_text("notes.txt", b"Hello\n\nworld") == "Hello\n\nworld"


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("notes.docx", b"content")


def test_extract_text_rejects_empty_text():
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("empty.txt", b"   \n  ")
