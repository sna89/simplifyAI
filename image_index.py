import os
import json
import hashlib
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.models.Collection import Collection
from sparse_search import ensure_sparse_index
import logging
from sparse_search import build_and_save_bm25_index


# Locations and defaults
INDEX_DIR = os.path.join("pecs_index")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma")
COLLECTION_NAME = "pecs_images"
DATA_METADATA_JSON = os.path.join("dataset", "metadata.json")


def _list_image_files(root_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn.lower())[1] in exts:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def _encode_images(model: SentenceTransformer, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    images: List[Image.Image] = []
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        images.append(img)
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    emb = model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def _path_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def _load_image_id_to_word(meta_path: str) -> Dict[str, str]:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("image_id_to_word", {}) or {}
    except Exception:
        return {}


def _metadatas_for_paths(image_paths: List[str], image_id_to_word: Dict[str, str]) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for p in image_paths:
        fname = os.path.basename(p)
        image_id = os.path.splitext(fname)[0]
        metas.append({
            "path": p,
            "image_id": image_id,
            "word": image_id_to_word.get(image_id)
        })
    return metas


def get_collection() -> Collection:
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def populate_index(image_dir: str, model: SentenceTransformer, force: bool = False) -> Collection:
    os.makedirs(INDEX_DIR, exist_ok=True)
    coll = get_collection()

    # If already populated and not forced, skip
    try:
        if not force and coll.count() and coll.count() > 0:
            return coll
    except Exception:
        pass

    image_paths = _list_image_files(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    embs = _encode_images(model, image_paths)
    ids = [_path_id(p) for p in image_paths]
    image_id_to_word = _load_image_id_to_word(DATA_METADATA_JSON)
    metadatas = _metadatas_for_paths(image_paths, image_id_to_word)

    chunk = 1024
    for start in range(0, len(ids), chunk):
        coll.upsert(
            ids=ids[start:start+chunk],
            embeddings=embs[start:start+chunk].tolist(),
            metadatas=metadatas[start:start+chunk]
        )
    return coll


def ensure_index(image_dir: str, model: SentenceTransformer, force: bool = False) -> Collection:
    """Ensures both dense (Chroma) and sparse (BM25) indexes are populated."""
    # First, ensure the sparse index is ready, as it depends on the metadata file.
    ensure_sparse_index()
    
    # Now, proceed with the dense index.
    if force:
        return populate_index(image_dir, model, force=True)
    coll = get_collection()
    try:
        if not coll.count() or coll.count() == 0:
            return populate_index(image_dir, model, force=False)
        return coll
    except Exception:
        return populate_index(image_dir, model, force=False)


