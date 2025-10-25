import os

# --- Model Names ---
TEXT_EM_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMG_EM_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-he-en"
LLM_MODEL_NAME = "google/flan-t5-base"

# --- File Paths and Directories ---
IMAGE_DIR = os.path.join("dataset", "images")
DATA_METADATA_JSON = os.path.join("dataset", "metadata.json")
IMAGE_EXTENSION = ".png"

# --- Indexing and Matching Parameters ---
TOP_K = 6
DISTANCE_THRESHOLD = 1.0  # Cosine distance, lower is better. Max is 2.0.

# --- Query Reformulation ---
REFORMULATION_PROMPT = """You are a helpful assistant that reformulates English sentences.

TASK: Given an English sentence, create 5 different ways to express it using different words, phrases, or sentence structures. 
Focus on synonyms, alternative expressions, and simpler language while maintaining the core meaning. 
Provide exactly 5 reformulations, one per line.
Make sure that each sentence is different from the original sentence and from the other reformulations.

---
INPUT: "I want to eat an apple."
OUTPUT:
I would like to have an apple.
Can I get an apple to eat?
Feeling hungry for an apple.
Time for an apple snack.
Give me an apple.
---
INPUT: "Let's go to the park to play."
OUTPUT:
I want to play at the park.
Can we go to the playground?
Let's have fun at the park.
It's time to visit the park.
We should go outside and play.
---
INPUT: "{input_text}"
OUTPUT:
"""
