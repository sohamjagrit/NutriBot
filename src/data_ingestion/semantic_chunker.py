"""Semantic chunking for nutritional documents.

Uses markdown structure to identify and preserve semantic units:
- Headings (sections, subsections)
- Paragraphs
- Lists
- Tables
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk."""
    text: str
    heading: str = ""
    block_type: str = "paragraph"  # paragraph, heading, list, table
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


class SemanticChunker:
    """Chunk markdown respecting document structure."""

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        """Initialize chunker.

        Args:
            max_chunk_size: Maximum words per chunk
            min_chunk_size: Minimum words before merging with next block
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, markdown: str) -> List[SemanticChunk]:
        """Chunk markdown by semantic structure.

        Strategy:
        1. Parse markdown into blocks (headings, paragraphs, lists, tables)
        2. Group blocks into chunks respecting:
           - Keep headings with their content
           - Don't split tables/lists
           - Merge small blocks with next larger one
        3. Respect max_chunk_size by splitting long paragraphs only

        Args:
            markdown: Markdown content from Docling

        Returns:
            List of SemanticChunk objects
        """
        blocks = self._parse_blocks(markdown)
        chunks = self._merge_blocks(blocks)
        chunks = self._split_large_chunks(chunks)
        return chunks

    def _parse_blocks(self, markdown: str) -> List[Tuple[str, str, str]]:
        """Parse markdown into semantic blocks.

        Returns:
            List of (block_type, heading, text) tuples
        """
        blocks = []
        lines = markdown.split('\n')
        current_block = []
        current_heading = ""
        current_type = "paragraph"

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect headings
            if line.startswith('#'):
                # Save previous block
                if current_block:
                    text = '\n'.join(current_block).strip()
                    if text:
                        blocks.append((current_type, current_heading, text))
                    current_block = []

                # Extract heading level and text
                heading_level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()

                blocks.append(("heading", heading_text, line.strip()))
                current_heading = heading_text
                current_type = "paragraph"

            # Detect tables (start with |)
            elif line.strip().startswith('|'):
                # Save current block
                if current_block:
                    text = '\n'.join(current_block).strip()
                    if text:
                        blocks.append((current_type, current_heading, text))
                    current_block = []

                # Collect entire table
                table_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                i -= 1

                table_text = '\n'.join(table_lines)
                blocks.append(("table", current_heading, table_text))

            # Detect lists
            elif line.strip() and (line.lstrip().startswith('-') or line.lstrip().startswith('*')):
                # Save current block
                if current_block:
                    text = '\n'.join(current_block).strip()
                    if text:
                        blocks.append((current_type, current_heading, text))
                    current_block = []

                # Collect entire list
                list_lines = [line]
                i += 1
                while i < len(lines) and (
                    lines[i].lstrip().startswith('-') or
                    lines[i].lstrip().startswith('*') or
                    (lines[i].startswith('  ') and lines[i].strip())
                ):
                    list_lines.append(lines[i])
                    i += 1
                i -= 1

                list_text = '\n'.join(list_lines)
                blocks.append(("list", current_heading, list_text))

            # Regular paragraph
            elif line.strip():
                current_block.append(line)

            # Empty line (paragraph break)
            elif current_block:
                text = '\n'.join(current_block).strip()
                if text:
                    blocks.append((current_type, current_heading, text))
                current_block = []

            i += 1

        # Save remaining block
        if current_block:
            text = '\n'.join(current_block).strip()
            if text:
                blocks.append((current_type, current_heading, text))

        return blocks

    def _merge_blocks(self, blocks: List[Tuple[str, str, str]]) -> List[SemanticChunk]:
        """Merge small blocks with adjacent content.

        - Keep headings with their content
        - Don't split tables/lists
        - Merge tiny paragraphs into larger chunks
        """
        merged = []
        i = 0

        while i < len(blocks):
            block_type, heading, text = blocks[i]

            # Headings: always group with following content
            if block_type == "heading":
                chunk_text = text
                i += 1

                # Add following blocks until we hit another heading or max size
                while i < len(blocks):
                    next_type, next_heading, next_text = blocks[i]

                    if next_type == "heading":
                        break

                    chunk_text += "\n\n" + next_text

                    # Stop if we reach max size
                    if len(chunk_text.split()) >= self.max_chunk_size:
                        break

                    i += 1

                merged.append(SemanticChunk(
                    text=chunk_text,
                    heading=heading,
                    block_type="section"
                ))

            # Tables & lists: keep intact, don't merge
            elif block_type in ("table", "list"):
                merged.append(SemanticChunk(
                    text=text,
                    heading=heading,
                    block_type=block_type
                ))
                i += 1

            # Small paragraphs: merge with next
            else:
                word_count = len(text.split())

                if word_count < self.min_chunk_size and i + 1 < len(blocks):
                    # Merge with next block
                    chunk_text = text
                    i += 1

                    while i < len(blocks):
                        next_type, next_heading, next_text = blocks[i]

                        if next_type == "heading":
                            break

                        chunk_text += "\n\n" + next_text
                        i += 1

                        if len(chunk_text.split()) >= self.max_chunk_size:
                            break

                    merged.append(SemanticChunk(
                        text=chunk_text,
                        heading=heading,
                        block_type="paragraph"
                    ))
                else:
                    # Paragraph is large enough
                    merged.append(SemanticChunk(
                        text=text,
                        heading=heading,
                        block_type="paragraph"
                    ))
                    i += 1

        return merged

    def _split_large_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Split chunks that exceed max_chunk_size.

        Only splits regular paragraphs, keeps tables/lists intact.
        """
        result = []

        for chunk in chunks:
            if chunk.word_count > self.max_chunk_size and chunk.block_type == "paragraph":
                # Split by sentences or words
                sentences = self._split_sentences(chunk.text)
                sub_chunks = self._group_sentences(sentences, chunk.heading)
                result.extend(sub_chunks)
            else:
                result.append(chunk)

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(self, sentences: List[str], heading: str) -> List[SemanticChunk]:
        """Group sentences into chunks."""
        chunks = []
        current_chunk = []
        current_words = 0

        for sentence in sentences:
            words = len(sentence.split())

            if current_words + words > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    heading=heading,
                    block_type="paragraph"
                ))
                current_chunk = [sentence]
                current_words = words
            else:
                current_chunk.append(sentence)
                current_words += words

        # Save remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(SemanticChunk(
                text=chunk_text,
                heading=heading,
                block_type="paragraph"
            ))

        return chunks


def test_chunker():
    """Test semantic chunker."""
    test_md = """# Nutrition Basics

Nutrition is the science of food and its effects on living organisms.

## Macronutrients

Macronutrients are nutrients needed in large quantities.

### Proteins

Proteins are made of amino acids. They are essential for building muscle and other tissues.

### Carbohydrates

Carbohydrates provide energy to the body.

## Micronutrients

Micronutrients are needed in smaller quantities but are vital for health.

| Mineral | Function | Sources |
|---------|----------|---------|
| Iron | Oxygen transport | Red meat, spinach |
| Calcium | Bone health | Dairy, leafy greens |

- Zinc supports immunity
- Magnesium aids muscle function
- Potassium regulates blood pressure
"""

    chunker = SemanticChunker(max_chunk_size=150, min_chunk_size=50)
    chunks = chunker.chunk(test_md)

    print("Semantic Chunks:")
    print("=" * 70)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{i}. [{chunk.block_type}] {chunk.heading}")
        print(f"   Words: {chunk.word_count}")
        print(f"   Text: {chunk.text[:100]}...")


if __name__ == "__main__":
    test_chunker()
