import logging
import argparse
import config
from models import load_model
from image_index import ensure_index
from matching import match_images_with_fusion
from visualization import plot_results_grid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(query: str, plot: bool = True, use_fusion: bool = True, force_reindex: bool = False):
    """
    Main entry point for the multimodal matching pipeline.
    """
    logging.info("--- Starting Multimodal Matching Pipeline ---")
    
    # Load models
    text_emb_model = load_model(config.TEXT_EM_MODEL_NAME)
    img_emb_model = load_model(config.IMG_EM_MODEL_NAME)
    
    # Ensure the image index is populated
    logging.info("Ensuring image index is populated...")
    collection = ensure_index(config.IMAGE_DIR, img_emb_model, force=force_reindex)
    
    # Perform matching
    results = match_images_with_fusion(
        query,
        text_emb_model,
        collection,
        top_k=config.TOP_K,
        max_distance=config.DISTANCE_THRESHOLD,
        use_reformulation=use_fusion
    )
    
    # Print and plot results
    if results and results != ["NONE"]:
        logging.info(f"Displaying {len(results)} results for query '{query}':")
        for i, result in enumerate(results):
            print(f"{i+1}. Path: {result.get('path')} | Fusion Score: {result.get('fusion_score', 'N/A'):.4f} | Matched Queries: {result.get('num_queries_matched')}")
    else:
        logging.info(f"No results found for query '{query}'.")
        print("['NONE']")

    if plot:
        plot_results_grid(results)
        
    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal PECS Image Search")
    parser.add_argument("--query", type=str, default="ילד משחק בכדור", help="The Hebrew sentence to search for. Default: 'ילד משחק בכדור'")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable displaying the results in a plot.")
    parser.add_argument("--no-fusion", action="store_false", dest="use_fusion", help="Disable query reformulation and fusion ranking.")
    parser.add_argument("--force-reindex", action="store_true", help="Force rebuild of the image index.")
    
    # Set default values for boolean flags that should be True by default
    parser.set_defaults(plot=True, use_fusion=True)
    
    args = parser.parse_args()
    
    main(query=args.query, plot=args.plot, use_fusion=args.use_fusion, force_reindex=args.force_reindex)
