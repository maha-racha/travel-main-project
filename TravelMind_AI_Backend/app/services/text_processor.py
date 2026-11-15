import tiktoken
from typing import List
from app.core.config import Settings


class TextProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []

        if len(tokens) <= self.chunk_size:
            return [text]

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            if i + self.chunk_size >= len(tokens):
                break

        return chunks
