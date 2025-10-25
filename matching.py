import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from transformers import pipeline
from text_processing import preprocess_hebrew, reformulate_query_with_llm, load_llm_for_reformulation
from sparse_search import load_bm25_index, search_sparse
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """
    Performs Reciprocal Rank Fusion (RRF) on multiple ranked lists of document IDs.
    Returns a dictionary of doc_id -> RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return rrf_scores

def match_images_with_fusion(
    hebrew_sentence: str,
    model: SentenceTransformer,
    collection: Collection,
    top_k: int = 3,
    max_distance: float = 0.25,
    use_reformulation: bool = True
) -> List[Dict[str, Any]] | List[str]:
    
    logging.info(f"Starting image match for query: '{hebrew_sentence}'")
    
    # 1) Encode original Hebrew
    hebrew_sentence = preprocess_hebrew(hebrew_sentence)
    logging.info("Encoding Hebrew query...")
    query_emb_heb = model.encode([hebrew_sentence], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    # 2) Translate to English and encode
    logging.info("Translating Hebrew to English...")
    translator = pipeline("translation", model=config.TRANSLATION_MODEL_NAME)
    translation = translator(hebrew_sentence)
    en_text = translation[0].get("translation_text", "") if translation else ""
    logging.info(f"Translated text: '{en_text}'. Encoding English query...")
    query_emb_en = model.encode([en_text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    # 3) Generate and encode reformulated queries if enabled
    reformulated_embeddings = []
    if use_reformulation and en_text:
        try:
            tokenizer, llm_model = load_llm_for_reformulation()
            reformulations = reformulate_query_with_llm(en_text, tokenizer, llm_model)
            logging.info("Encoding reformulated queries...")
            for reformulation in reformulations:
                if reformulation and reformulation != en_text:
                    emb = model.encode([reformulation], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                    reformulated_embeddings.append(emb)
        except Exception as e:
            logging.warning(f"LLM reformulation failed: {e}. Continuing without it.")

    # 4) Query Chroma with all embeddings
    all_queries = [query_emb_heb, query_emb_en] + reformulated_embeddings
    all_query_names = ["Hebrew", "English"] + [f"Reform_{i+1}" for i in range(len(reformulated_embeddings))]
    all_candidates: Dict[str, Dict[str, Any]] = {}
    dense_results_by_query: Dict[str, List[str]] = {}
    
    logging.info(f"Querying dense index with {len(all_queries)} embeddings...")
    for query_emb, query_name in zip(all_queries, all_query_names):
        if query_emb.size == 0: continue
        res = collection.query(
            query_embeddings=query_emb.tolist(), 
            n_results=top_k * 2,
            include=["metadatas", "distances"]
        )
        if not res or not res.get("ids"): continue
        
        ids, dists, metas = (res.get(k, [[]])[0] or [] for k in ["ids", "distances", "metadatas"])
        
        current_query_ranked_ids = []
        for rid, dist, meta in zip(ids, dists, metas):
            if str(rid) not in all_candidates:
                all_candidates[str(rid)] = {
                    "id": str(rid), "meta": meta, "query_matches": {},
                    "best_distance": float('inf'), "total_distance": 0.0, "num_queries_matched": 0
                }
            if float(dist) < all_candidates[str(rid)]["best_distance"]:
                all_candidates[str(rid)]["best_distance"] = float(dist)
            all_candidates[str(rid)]["query_matches"][query_name] = float(dist)
            all_candidates[str(rid)]["total_distance"] += float(dist)
            all_candidates[str(rid)]["num_queries_matched"] += 1
            current_query_ranked_ids.append(str(rid))

        dense_results_by_query[query_name] = current_query_ranked_ids

    # 4.5) Perform Sparse Search
    bm25_index, doc_ids = load_bm25_index()
    sparse_results = []
    if bm25_index and doc_ids:
        sparse_results_tuples = search_sparse(bm25_index, doc_ids, en_text, top_k=top_k * 2)
        sparse_ranked_ids = [doc_id for doc_id, score in sparse_results_tuples]
        dense_results_by_query["Sparse_BM25"] = sparse_ranked_ids
        
        # Also add sparse results to the main candidate pool to ensure metadata is available
        for image_id, score in sparse_results_tuples:
            if image_id not in all_candidates:
                # Note: We don't have dense metadata here, but RRF only needs the ID.
                # We can retrieve metadata later if needed.
                all_candidates[image_id] = {"id": image_id, "meta": {"image_id": image_id}}

    if not all_candidates:
        logging.warning("No candidates found after querying.")
        return ["NONE"]

    # 5) Reciprocal Rank Fusion
    logging.info("Performing Reciprocal Rank Fusion on dense and sparse results...")
    ranked_lists = list(dense_results_by_query.values())
    rrf_scores = reciprocal_rank_fusion(ranked_lists)

    # 6) Sort by RRF score and format results
    sorted_candidates_by_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    results = []
    for doc_id, rrf_score in sorted_candidates_by_rrf:
        if doc_id in all_candidates:
            cand = all_candidates[doc_id]
            meta = cand.get("meta") or {}
            
            # Use a placeholder if best_distance wasn't calculated (from sparse-only results)
            best_dist = cand.get("best_distance", 1.0) 
            
            if best_dist <= max_distance:
                results.append({
                    "id": doc_id,
                    "path": meta.get("path"), "word": meta.get("word"), "image_id": meta.get("image_id"),
                    "rrf_score": rrf_score,
                    "best_distance": best_dist,
                    "num_queries_matched": cand.get("num_queries_matched", 0)
                })
                if len(results) >= top_k: break
    
    logging.info(f"Found {len(results)} final results after RRF and filtering.")
    return results if results else ["NONE"]
