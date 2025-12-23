import json
import os
from typing import List, Dict, Any

INPUT_FILE = "scraped_data.json"
OUTPUT_FILE = "processed_chunks.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_data(filepath: str) -> List[Dict[str, Any]]:
    """Loads raw scraped data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_chunks(content: str, metadata: Dict[str, str]) -> List[Dict[str, Any]]:
    """Splits a string into overlapping chunks with metadata."""
    chunks: List[Dict[str, Any]] = []
    start = 0
    content_length = len(content)

    while start < content_length:
        end = start + CHUNK_SIZE
        chunk_text = content[start:end]
        
        chunk_metadata = {
            "text": chunk_text,
            "source_url": metadata["url"],
            "source_title": metadata["title"],
            "chunk_index": len(chunks)
        }
        chunks.append(chunk_metadata)

        # Move the window forward by CHUNK_SIZE - CHUNK_OVERLAP
        start += CHUNK_SIZE - CHUNK_OVERLAP
    
    return chunks

def process_data() -> List[Dict[str, Any]]:
    """Loads raw data, chunks it, and returns the list of processed chunks."""
    print("Starting data processing...")
    raw_data = load_data(INPUT_FILE)
    if not raw_data:
        print("No raw data to process. Run scraper.py first.")
        return []

    processed_chunks: List[Dict[str, Any]] = []
    
    for item in raw_data:
        content = item.get("content", "")
        metadata = {"url": item.get("url", "N/A"), "title": item.get("title", "N/A")}
        if content:
            new_chunks = create_chunks(content, metadata)
            processed_chunks.extend(new_chunks)

    print("Processing complete.")
    return processed_chunks

def save_chunks(chunks: List[Dict[str, Any]]):
    """Saves the processed chunks to a JSON file."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)
    print(f"Processed chunks successfully saved to {OUTPUT_FILE}")

def print_stats(chunks: List[Dict[str, Any]]):
    """Prints statistics about the processed data."""
    total_chunks = len(chunks)
    unique_sources = len(set(chunk["source_url"] for chunk in chunks))
    
    if total_chunks > 0:
        avg_length = sum(len(chunk["text"]) for chunk in chunks) / total_chunks
    else:
        avg_length = 0

    print("\n--- Data Processing Statistics ---")
    print(f"Total Chunks Created: {total_chunks}")
    print(f"Unique Source URLs: {unique_sources}")
    print(f"Average Chunk Length: {avg_length:.2f} characters")
    print("----------------------------------")

if __name__ == "__main__":
    chunks = process_data()
    if chunks:
        save_chunks(chunks)
        print_stats(chunks)