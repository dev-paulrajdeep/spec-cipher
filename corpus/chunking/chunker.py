"""Create configurable chunks from detected sections without embeddings or retrieval logic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from corpus.models import DocumentChunk, Section

CHUNKING_VERSION = "1.1"


@dataclass(frozen=True)
class ChunkingConfig:
    """Character-based chunk limits, chosen to be tokenizer-independent and deterministic."""

    max_chars: int = 2_000
    overlap_chars: int = 200

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than max_chars")


class DocumentChunker:
    """Chunk within a section first and split paragraphs only when necessary."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(
        self, sections: tuple[Section, ...], standard_number: str | None, source_file: str
    ) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            for part in self._split(section.text):
                ordinal = len(chunks) + 1
                identity = "|".join(
                    (source_file, standard_number or "", section.section_number or "preamble", str(ordinal))
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id="chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                        standard_number=standard_number,
                        source_file=source_file,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_number=section.section_number,
                        section_title=section.title,
                        text=part,
                        chunking_version=CHUNKING_VERSION,
                    )
                )
        return tuple(chunks)

    def _split(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in (item.strip() for item in re.split(r"\n{2,}", text) if item.strip()):
            units.extend(self._split_oversize(paragraph))
        if not units and text.strip():
            units = self._split_oversize(text.strip())
        chunks: list[str] = []
        current = ""
        for unit in units:
            proposed = unit if not current else f"{current}\n\n{unit}"
            if current and len(proposed) > self.config.max_chars:
                chunks.append(current)
                current = self._overlap(current)
                proposed = unit if not current else f"{current}\n\n{unit}"
            if len(proposed) > self.config.max_chars:
                # This can only be an overlap-plus-unit edge. Do not emit a
                # retrieval-meaningless chunk containing just the overlap.
                current = unit
            else:
                current = proposed
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _split_oversize(self, text: str) -> list[str]:
        if len(text) <= self.config.max_chars:
            return [text]
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
        if len(sentences) > 1:
            result: list[str] = []
            current = ""
            for sentence in sentences:
                if current and len(current) + 1 + len(sentence) > self.config.max_chars:
                    result.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                result.append(current)
            return result
        return [text[index : index + self.config.max_chars].strip() for index in range(0, len(text), self.config.max_chars)]

    def _overlap(self, text: str) -> str:
        if not self.config.overlap_chars:
            return ""
        candidate = text[-self.config.overlap_chars :].strip()
        boundary = candidate.find(" ")
        return candidate[boundary + 1 :] if boundary >= 0 else candidate
