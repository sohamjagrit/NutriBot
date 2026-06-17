#!/usr/bin/env python3
"""SageMaker Processing Job script for Docling PDF parsing.

This script runs inside a SageMaker Processing Job container.
It reads PDFs from S3, parses them with Docling, and uploads results back to S3.

Environment variables:
  - AWS_REGION: AWS region
  - S3_BUCKET_NAME: S3 bucket name
  - RAW_PREFIX: S3 prefix for raw PDFs (e.g., "raw/pdfs")
  - PARSED_PREFIX: S3 prefix for parsed output (e.g., "parsed")
  - CHUNK_SIZE: Chunk size in tokens (default: 512)
  - CHUNK_OVERLAP: Chunk overlap (default: 50)
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data_ingestion.semantic_chunker import SemanticChunker

try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    print("Warning: Docling not installed")
    HAS_DOCLING = False
    DocumentConverter = None

# Simple logging
def log(level: str, msg: str):
    """Log message with timestamp."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {msg}", flush=True)


@dataclass
class DocumentChunk:
    """Represents a parsed document chunk."""

    chunk_id: str
    source_file: str
    source_type: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    document_title: Optional[str] = None
    source_url: Optional[str] = None
    table_data: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProcessingJobParser:
    """PDF parser for SageMaker Processing Jobs."""

    def __init__(self, s3_client, bucket: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize parser."""
        self.s3_client = s3_client
        self.bucket = bucket
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = DocumentConverter() if HAS_DOCLING else None
        self.semantic_chunker = SemanticChunker(max_chunk_size=chunk_size, min_chunk_size=chunk_size // 4)

        self.stats = {
            "pdfs_found": 0,
            "pdfs_parsed": 0,
            "pdfs_failed": 0,
            "total_pages": 0,
            "total_pdf_size_mb": 0.0,
            "total_chunks": 0,
            "parsing_time_sec": 0.0,
            "upload_time_sec": 0.0,
        }

        log("INFO", f"Parser initialized (docling={HAS_DOCLING})")

    def list_pdfs_from_s3(self, prefix: str) -> List[str]:
        """List all PDFs in S3 prefix."""
        log("INFO", f"Listing PDFs from s3://{self.bucket}/{prefix}")

        pdf_keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    if obj["Key"].endswith(".pdf"):
                        pdf_keys.append(obj["Key"])
                        self.stats["total_pdf_size_mb"] += obj["Size"] / (1024 * 1024)

            log("INFO", f"Found {len(pdf_keys)} PDFs")
            self.stats["pdfs_found"] = len(pdf_keys)
            return pdf_keys

        except Exception as e:
            log("ERROR", f"Failed to list PDFs: {e}")
            return []

    def parse_pdf_from_s3(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Parse PDF from S3."""
        log("INFO", f"Parsing {Path(s3_key).name}...")

        try:
            # Download PDF from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            pdf_bytes = response["Body"].read()

            # Write to temp file (Docling needs file path)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                # Parse with Docling
                result = self.converter.convert(tmp_path)
                document = result.document

                # Export to markdown
                markdown = document.export_to_markdown()

                # Get page count
                page_count = len(document.pages) if hasattr(document, "pages") else 0
                self.stats["total_pages"] += page_count

                log("INFO", f"  Parsed: {page_count} pages, {len(markdown)} chars")

                return {
                    "markdown": markdown,
                    "page_count": page_count,
                }

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            log("ERROR", f"  Failed: {e}")
            self.stats["pdfs_failed"] += 1
            return None

    def upload_markdown_to_s3(
        self, s3_key: str, markdown: str, output_prefix: str
    ) -> Optional[str]:
        """Upload markdown to S3."""
        try:
            pdf_name = Path(s3_key).stem
            markdown_key = f"{output_prefix}/markdown/{pdf_name}.md"

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=markdown_key,
                Body=markdown.encode("utf-8"),
                ContentType="text/markdown",
            )

            log("INFO", f"  Uploaded markdown: {len(markdown)} chars")
            return markdown_key

        except Exception as e:
            log("ERROR", f"  Failed to upload markdown: {e}")
            return None

    def chunk_markdown(
        self,
        markdown: str,
        pdf_name: str,
        source_type: str,
        page_count: int,
    ) -> List[DocumentChunk]:
        """Chunk markdown content using semantic structure."""
        if not markdown.strip():
            log("WARNING", f"Empty markdown for {pdf_name}")
            return []

        # Use semantic chunking that respects document structure
        semantic_chunks = self.semantic_chunker.chunk(markdown)

        document_chunks = []
        for i, sem_chunk in enumerate(semantic_chunks):
            chunk_id = f"{Path(pdf_name).stem}_{i:03d}"
            token_count = len(sem_chunk.text.split())

            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                source_file=pdf_name,
                source_type=source_type,
                content=sem_chunk.text,
                chunk_index=i,
                total_chunks=len(semantic_chunks),
                token_count=token_count,
                section_title=sem_chunk.heading or None,
                document_title=Path(pdf_name).stem.replace("_", " ").title(),
                metadata={
                    "source": pdf_name,
                    "source_type": source_type,
                    "total_pages": page_count,
                    "block_type": sem_chunk.block_type,
                    "section": sem_chunk.heading,
                },
            )
            document_chunks.append(doc_chunk)

        return document_chunks

    def determine_source_type(self, s3_key: str) -> str:
        """Determine document type from S3 key."""
        if "research_papers" in s3_key:
            return "research_paper"
        elif "fda_labels" in s3_key:
            return "fda_label"
        elif "dietary_guidelines" in s3_key:
            return "dietary_guideline"
        else:
            return "unknown"

    def process_all_pdfs(
        self, input_prefix: str, output_prefix: str
    ) -> List[DocumentChunk]:
        """Process all PDFs."""
        log("INFO", "=" * 70)
        log("INFO", "Processing PDFs from S3")
        log("INFO", "=" * 70)

        start_time = time.time()
        all_chunks = []

        # List PDFs
        pdf_keys = self.list_pdfs_from_s3(input_prefix)
        if not pdf_keys:
            log("ERROR", "No PDFs found")
            return all_chunks

        # Process each PDF
        for idx, pdf_key in enumerate(pdf_keys, 1):
            log("INFO", f"\n[{idx}/{len(pdf_keys)}] Processing {Path(pdf_key).name}")

            # Parse PDF
            parsed = self.parse_pdf_from_s3(pdf_key)
            if not parsed:
                continue

            # Upload markdown
            self.upload_markdown_to_s3(pdf_key, parsed["markdown"], output_prefix)

            # Chunk markdown
            source_type = self.determine_source_type(pdf_key)
            chunks = self.chunk_markdown(
                parsed["markdown"],
                Path(pdf_key).name,
                source_type,
                parsed["page_count"],
            )

            if chunks:
                all_chunks.extend(chunks)
                self.stats["pdfs_parsed"] += 1
                self.stats["total_chunks"] += len(chunks)
                log("INFO", f"  Created {len(chunks)} chunks")

        self.stats["parsing_time_sec"] = time.time() - start_time

        return all_chunks

    def upload_chunks_to_s3(
        self, chunks: List[DocumentChunk], output_prefix: str
    ) -> bool:
        """Upload chunks to S3."""
        if not chunks:
            log("WARNING", "No chunks to upload")
            return False

        log("INFO", f"\nUploading {len(chunks)} chunks to S3")

        try:
            upload_start = time.time()

            # Convert to JSONL
            jsonl_lines = [json.dumps(asdict(chunk), default=str) for chunk in chunks]
            jsonl_content = "\n".join(jsonl_lines)

            # Upload to S3
            s3_key = f"{output_prefix}/chunks/parsed_chunks.jsonl"

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=jsonl_content.encode("utf-8"),
                ContentType="application/jsonl",
            )

            self.stats["upload_time_sec"] = time.time() - upload_start
            log("INFO", f"Uploaded {len(chunks)} chunks to s3://{self.bucket}/{s3_key}")
            return True

        except Exception as e:
            log("ERROR", f"Failed to upload chunks: {e}")
            return False

    def print_summary(self):
        """Print execution summary."""
        log("INFO", "\n" + "=" * 70)
        log("INFO", "SUMMARY")
        log("INFO", "=" * 70)
        log("INFO", f"PDFs found:   {self.stats['pdfs_found']}")
        log("INFO", f"PDFs parsed:  {self.stats['pdfs_parsed']}")
        log("INFO", f"PDFs failed:  {self.stats['pdfs_failed']}")
        log("INFO", f"Total pages:  {self.stats['total_pages']}")
        log("INFO", f"Total size:   {self.stats['total_pdf_size_mb']:.1f} MB")
        log("INFO", f"Total chunks: {self.stats['total_chunks']}")
        log("INFO", f"Parse time:   {self.stats['parsing_time_sec']:.1f}s")
        log("INFO", f"Upload time:  {self.stats['upload_time_sec']:.1f}s")
        log("INFO", "=" * 70)


def main():
    """Main entry point."""
    # Get configuration from environment variables
    bucket = os.environ.get("S3_BUCKET_NAME", "nutrition-usda-foods")
    region = os.environ.get("AWS_REGION", "us-east-1")
    raw_prefix = os.environ.get("RAW_PREFIX", "raw/pdf's")
    parsed_prefix = os.environ.get("PARSED_PREFIX", "parsed")
    chunk_size = int(os.environ.get("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "50"))

    log("INFO", f"Configuration:")
    log("INFO", f"  Bucket: {bucket}")
    log("INFO", f"  Region: {region}")
    log("INFO", f"  Input prefix: {raw_prefix}")
    log("INFO", f"  Output prefix: {parsed_prefix}")
    log("INFO", f"  Chunk size: {chunk_size}")
    log("INFO", f"  Chunk overlap: {chunk_overlap}")

    # Initialize S3
    log("INFO", "Initializing S3...")
    s3_client = boto3.client("s3", region_name=region)

    # Initialize parser
    parser = ProcessingJobParser(s3_client, bucket, chunk_size, chunk_overlap)

    # Process PDFs
    chunks = parser.process_all_pdfs(raw_prefix, parsed_prefix)

    if not chunks:
        log("ERROR", "No chunks generated")
        return False

    # Upload chunks
    if not parser.upload_chunks_to_s3(chunks, parsed_prefix):
        log("ERROR", "Failed to upload chunks")
        return False

    # Print summary
    parser.print_summary()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        log("ERROR", f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
