import logging
import torch
from typing import List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_hebrew(text: str) -> str:
    """Preprocesses Hebrew text by stripping whitespace."""
    logging.debug(f"Preprocessing Hebrew text: '{text}'")
    return text.strip()


def load_llm_for_reformulation() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Loads a lightweight LLM and its tokenizer for query reformulation."""
    logging.info(f"Loading LLM for query reformulation: {config.LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL_NAME)

    logging.info("LLM loaded successfully.")
    return tokenizer, model


def reformulate_query_with_llm(english_text: str, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> List[str]:
    """Uses LLM to generate 5 query reformulations."""
    prompt = config.REFORMULATION_PROMPT.format(input_text=english_text)
    logging.info(f"Reformulating query for: '{english_text}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate reformulations using sampling for more creative results
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            num_return_sequences=5,
            do_sample=True,
            temperature=1,  # Adjust for creativity; lower is more deterministic
            top_p=0.95,       # Adjust nucleus sampling probability
        )
    
    # Decode and extract unique reformulations
    all_generated_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
          
    logging.info(f"Generated {len(all_generated_texts)} reformulations: {all_generated_texts}")
    return all_generated_texts
