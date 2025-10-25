import os
import json
from collections import defaultdict
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Phase 0: Config
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CACHE_DIR = os.path.join(DATASET_DIR, "cache")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")
MAX_WORKERS = 8
IMAGE_EXTENSION = ".png"

# Phase 1: Create folders
def ensure_folders() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# Helpers
def construct_image_url(image_id: str) -> str:
    # Construct the URL from the image_id as specified
    return f"https://api.arasaac.org/v1/pictograms/{image_id}?download=true"


def download_image(url: str, out_path: str) -> bool:
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return False
        # Convert bytes to image and save
        img = Image.open(BytesIO(resp.content))
        img.save(out_path)
        return True
    except Exception:
        return False


def save_metadata(word_to_image_ids: dict, image_id_to_metadata: dict) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "word_to_image_ids": word_to_image_ids,
                "image_id_to_metadata": image_id_to_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _dataset_exists() -> bool:
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(METADATA_PATH):
        return False
    try:
        # Consider dataset present if at least one image file exists
        for name in os.listdir(IMAGES_DIR):
            if name.lower().endswith(IMAGE_EXTENSION):
                return True
        return False
    except Exception:
        return False


def load_cache() -> tuple[dict, dict]:
    if not os.path.exists(METADATA_PATH):
        return {}, {}
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("word_to_image_ids", {}), data.get(
                "image_id_to_metadata", {}
            )
    except Exception:
        return {}, {}


def get_arasaac_keyword_data(language_code: str = "en") -> list[dict]:
    """Fetches keyword data from ARASAAC API."""
    url = f"https://api.arasaac.org/api/keywords/{language_code}"
    try:
        print(f"Fetching keywords from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        print("Successfully fetched keywords.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching keywords from ARASAAC API: {e}")
        return []


def get_pictograms_for_keyword(
    keyword: str, language_code: str = "en"
) -> list[dict]:
    """Fetches pictograms for a given keyword from the ARASAAC search API."""
    encoded_keyword = quote(keyword, safe="")
    url = f"https://api.arasaac.org/api/pictograms/{language_code}/search/{encoded_keyword}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pictograms for keyword '{keyword}': {e}")
        return []


def process_keyword(
    keyword: str,
    word_to_image_ids: dict[str, list[str]],
    image_id_to_metadata: dict[str, dict],
    lock: threading.Lock,
):
    """Processes a single keyword to fetch and store pictogram data."""
    keyword = keyword.strip()
    if not keyword:
        return

    pictograms = get_pictograms_for_keyword(keyword)
    for pictogram in pictograms:
        if "_id" not in pictogram:
            continue
        image_id = str(pictogram["_id"])
        tags = pictogram.get("tags", [])
        categories = pictogram.get("categories", [])

        with lock:
            word_to_image_ids[keyword].append(image_id)
            if image_id not in image_id_to_metadata:
                image_id_to_metadata[image_id] = {
                    "word": keyword,
                    "categories": categories,
                    "tags": tags,
                }


def download_metadata(unique_keywords: list[str]) -> tuple[dict, dict]:
    """Downloads metadata for a list of keywords."""
    word_to_image_ids: dict[str, list[str]] = defaultdict(list)
    image_id_to_metadata: dict[str, dict] = {}
    lock = threading.Lock()

    print(f"Found {len(unique_keywords)} unique keywords. Fetching pictograms...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_keyword, keyword, word_to_image_ids, image_id_to_metadata, lock
            )
            for keyword in unique_keywords
        ]
        with tqdm(total=len(futures), desc="Processing keywords") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  # To raise exceptions if any occurred
                except Exception as e:
                    print(f"Error processing a keyword: {e}")
                pbar.update(1)
    return word_to_image_ids, image_id_to_metadata


def download_image_dataset(force: bool = False) -> tuple[dict, dict]:
    # Phase 1: Create folders
    ensure_folders()

    word_to_image_ids, image_id_to_metadata = {}, {}
    # Load metadata from cache if it exists and not forced
    if not force and os.path.exists(METADATA_PATH):
        word_to_image_ids, image_id_to_metadata = load_cache()

    # If cache is empty or force is True, download metadata
    if force or not image_id_to_metadata:
        keyword_data = get_arasaac_keyword_data()
        if not keyword_data:
            print("Could not retrieve keyword data. Aborting.")
            return {}, {}
        unique_keywords = sorted(list(set(word for word in keyword_data["words"])))
        word_to_image_ids, image_id_to_metadata = download_metadata(unique_keywords)

    # Prepare download tasks for items not on disk
    tasks: list[tuple[str, str, str]] = []  # (image_id, url, out_path)
    for image_id in image_id_to_metadata:
        out_path = os.path.join(IMAGES_DIR, f"{image_id}{IMAGE_EXTENSION}")
        if os.path.exists(out_path):
            continue

        url = construct_image_url(image_id)
        tasks.append((image_id, url, out_path))

    # Execute downloads concurrently with progress bar
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {
                executor.submit(download_image, url, out_path): (image_id)
                for (image_id, url, out_path) in tasks
            }
            with tqdm(
                total=len(future_to_item), desc="Downloading images", unit="img"
            ) as pbar:
                for future in as_completed(future_to_item):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"An error occurred during image download: {e}")
                    pbar.update(1)

    # Phase 4: save dict
    # Convert defaultdict to regular dict for JSON
    final_word_to_image_ids = {k: v for k, v in word_to_image_ids.items()}
    save_metadata(final_word_to_image_ids, image_id_to_metadata)

    return final_word_to_image_ids, image_id_to_metadata


if __name__ == "__main__":
    # Default behavior: do not force re-download.
    download_image_dataset(force=True)

