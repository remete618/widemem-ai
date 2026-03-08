import hashlib


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
