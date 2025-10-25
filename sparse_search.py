import logging
import os
import json
import pickle
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi
import config

# --- File Paths ---
SPARSE_INDEX_DIR = os.path.join("pecs_index", "sparse")
BM25_INDEX_PATH = os.path.join(SPARSE_INDEX_DIR, "bm25_index.pkl")
BM25_DOC_IDS_PATH = os.path.join(SPARSE_INDEX_DIR, "bm25_doc_ids.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_metadata() -> Dict[str, str]:
    """Loads the image_id_to_word mapping from the main metadata file."""
    if not os.path.exists(config.DATA_METADATA_JSON):
        logging.warning(f"Metadata file not found at {config.DATA_METADATA_JSON}, cannot build sparse index.")
        return {}
    try:
        with open(config.DATA_METADATA_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("image_id_to_word", {}) or {}
    except Exception as e:
        logging.error(f"Failed to load metadata for sparse index: {e}")
        return {}

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    # Ensure text is a string before lowercasing
    return str(text).lower().split()

def build_and_save_bm25_index(doc_ids: List[str], documents: List[str]):
    """Builds a BM25 index from a list of IDs and documents, then saves it."""
    logging.info("Building sparse (BM25) index from metadata...")
    os.makedirs(SPARSE_INDEX_DIR, exist_ok=True)
    
    if not doc_ids or not documents:
        logging.warning("Cannot build sparse index with empty documents or IDs.")
        return

    corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(corpus)

    # Save the index and the document ID mapping (now SHA1s)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DOC_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
        
    logging.info(f"BM25 index built and saved with {len(doc_ids)} documents.")

def load_bm25_index() -> Tuple[BM25Okapi | None, List[str] | None]:
    """Loads a pre-built BM25 index from disk."""
    if not os.path.exists(BM25_INDEX_PATH) or not os.path.exists(BM25_DOC_IDS_PATH):
        return None, None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(BM25_DOC_IDS_PATH, "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        logging.info("Loaded existing BM25 index from disk.")
        return bm25, doc_ids
    except Exception as e:
        logging.error(f"Failed to load BM25 index: {e}")
        return None, None

def search_sparse(bm25: BM25Okapi, doc_ids: List[str], query: str, top_k: int) -> List[Tuple[str, float]]:
    """Performs a search on the BM25 index."""
    logging.info(f"Performing sparse search for query: '{query}'")
    tokenized_query = _tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get top k results
    top_n_indices = doc_scores.argsort()[::-1][:top_k]
    
    results = []
    for i in top_n_indices:
        # Only include results with a positive score
        if doc_scores[i] > 0:
            results.append((doc_ids[i], doc_scores[i]))
            
    return results

def ensure_sparse_index():
    """Checks if the BM25 index exists. Does not build it here."""
    if not os.path.exists(BM25_INDEX_PATH):
        logging.info("Sparse index not found. Building now...")
        build_and_save_bm25_index()
    else:
        logging.info("Found existing sparse index.")

if __name__ == '__main__':
    # Example usage: build and then search
    build_and_save_bm25_index()
    bm25_index, doc_ids = load_bm25_index()
    if bm25_index and doc_ids:
        results = search_sparse(bm25_index, doc_ids, "boy playing ball", top_k=5)
        print("Sparse search results:", results)
