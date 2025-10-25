import logging
from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    """Loads a SentenceTransformer model."""
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name, device=device or None)
    logging.info(f"Model '{model_name}' loaded successfully.")
    return model
