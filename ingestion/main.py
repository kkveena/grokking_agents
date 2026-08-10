import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from safetensors.numpy import load_file

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "week4docs").strip()

FILES_DIR = Path("ingestion/files")
JSONL_PATH = FILES_DIR / "ai_index_report_2026.jsonl"
SAFETENSORS_PATH = FILES_DIR / "ai_index_report_2026.safetensors"

def ingest():
    print(f"Starting ingestion for {JSONL_PATH.name}...")
    
    # 1. Initialize Qdrant Client
    api_key = QDRANT_API_KEY if QDRANT_API_KEY else None
    client = QdrantClient(url=QDRANT_URL, api_key=api_key)
    
    # Ensure collection exists
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)
    
    if not exists:
        print(f"Creating collection {QDRANT_COLLECTION}...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    
    # 2. Load Embeddings
    print("Loading embeddings...")
    embeddings_dict = load_file(str(SAFETENSORS_PATH))
    embeddings = embeddings_dict["embeddings"]
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # 3. Load and process OCR data
    print("Processing OCR data...")
    pages_data = []
    
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            page_elements = json.loads(line)
            page_number = i + 1
            
            # Combine all text content from page elements
            text_contents = [
                el["content"] for el in page_elements 
                if el.get("content") is not None
            ]
            combined_text = "\n\n".join(text_contents).strip()
            
            # We subtract 1 from page_number because embeddings are 0-indexed
            # and page numbers start at 1
            embedding = embeddings[page_number - 1]
            
            pages_data.append({
                "page_number": page_number,
                "text": combined_text,
                "embedding": embedding.tolist()
            })
    
    print(f"Prepared {len(pages_data)} pages for upload.")
    
    # 4. Upload to Qdrant
    print("Uploading to Qdrant...")
    file_name = "ai_index_report_2026.pdf" # Use the original PDF name for metadata
    processed_at = datetime.utcnow().isoformat()
    
    ids = [i for i in range(len(pages_data))]
    payloads = [
        {
            "file_name": file_name,
            "page_number": p["page_number"],
            "text": p["text"],
            "processed_at": processed_at
        }
        for p in pages_data
    ]
    vectors = [p["embedding"] for p in pages_data]
    
    client.upload_collection(
        collection_name=QDRANT_COLLECTION,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        parallel=4
    )
    
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    ingest()
