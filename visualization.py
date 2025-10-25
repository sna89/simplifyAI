import logging
import os
from typing import List, Dict, Any, Tuple
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_results_grid(results: List[Dict[str, Any]], cols: int = 3, figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plots result images in a simple grid using matplotlib."""
    if not results or (isinstance(results, list) and results and results[0] == "NONE"):
        logging.info("No results to display.")
        return

    valid_results = [r for r in results if isinstance(r, dict) and r.get("path") and os.path.exists(r["path"])]
    if not valid_results:
        logging.warning("No valid image paths found in results to display.")
        return

    n = len(valid_results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Image Matching Results", fontsize=16)

    ax_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for i, ax in enumerate(ax_list):
        if i < n:
            item = valid_results[i]
            try:
                img = Image.open(item["path"]).convert("RGB")
                ax.imshow(img)
                
                title = item.get("word") or os.path.basename(item["path"])
                score_info = ""
                if "fusion_score" in item:
                    score_info = f"Fusion: {item['fusion_score']:.3f} ({item['num_queries_matched']}Q)"
                elif "distance" in item:
                    score_info = f"Dist: {item['distance']:.3f}"
                
                if score_info:
                    title = f"{title}\n{score_info}"
                ax.set_title(title, fontsize=9)
            except Exception as e:
                logging.error(f"Failed to plot image {item.get('path')}: {e}")
                ax.set_title("Error", fontsize=9)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
