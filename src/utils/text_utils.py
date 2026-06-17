"""Text processing utilities for NutriBot."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters (keep letters, numbers, spaces, common punctuation)
    text = re.sub(r'[^\w\s\.\,\?\!\-]', '', text)
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap

    return chunks


def extract_nutrition_facts(text: str) -> dict:
    """Extract nutrition facts from text (basic implementation).

    Args:
        text: Text containing nutrition information

    Returns:
        Dictionary of extracted nutrition facts
    """
    facts = {}

    # Protein
    protein_match = re.search(r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s+(?:of\s+)?protein', text, re.IGNORECASE)
    if protein_match:
        facts['protein_g'] = float(protein_match.group(1))

    # Carbs
    carbs_match = re.search(r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s+(?:of\s+)?carb', text, re.IGNORECASE)
    if carbs_match:
        facts['carbs_g'] = float(carbs_match.group(1))

    # Fat
    fat_match = re.search(r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s+(?:of\s+)?fat', text, re.IGNORECASE)
    if fat_match:
        facts['fat_g'] = float(fat_match.group(1))

    # Calories
    cal_match = re.search(r'(\d+)\s*(?:kcal|calories?)', text, re.IGNORECASE)
    if cal_match:
        facts['calories'] = int(cal_match.group(1))

    return facts
