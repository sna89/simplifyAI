# Max Impact Project

This project is a multimodal matching system designed to find the most relevant images for a given Hebrew text query. It leverages state-of-the-art models for text and image embeddings, translation, and query reformulation to provide accurate and context-aware results.

## Features

- **Multilingual Support**: Translates Hebrew queries to English to leverage powerful English-based models.
- **Query Reformulation**: Uses a language model to generate multiple variations of the query, improving recall.
- **Dense and Sparse Search**: Combines dense vector search (ChromaDB) with sparse keyword search (BM25) for robust matching.
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple search strategies to produce a single, high-quality ranked list.
- **Modular and Extensible**: The codebase is organized into logical modules for models, indexing, matching, and visualization.

## How It Works

1.  **Input**: The user provides a Hebrew sentence as a query.
2.  **Preprocessing**: The Hebrew text is preprocessed.
3.  **Translation**: The query is translated into English.
4.  **Query Reformulation**: The English query is reformulated into several variants using a T5 model.
5.  **Embedding**: The original Hebrew, translated English, and reformulated queries are all encoded into high-dimensional vectors using a CLIP-based SentenceTransformer model.
6.  **Hybrid Search**:
    *   **Dense Search**: The query embeddings are used to search a ChromaDB vector index of pre-encoded images.
    *   **Sparse Search**: The English text is used to search a BM25 index built from image metadata (words associated with each image).
7.  **Fusion**: The results from all dense and sparse searches are combined using Reciprocal Rank Fusion.
8.  **Output**: The top-ranked images are returned, along with their scores, and displayed in a grid.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd max_impact
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset**:
    Run the dataset download script for the first time. This will download the images and create the necessary metadata files.
    ```bash
    python dataset.py
    ```

## Usage

You can run the matching pipeline from the command line.

**Basic Usage**:
The default query is "ילד משחק בכדור" (A boy plays with a ball).

```bash
python main.py
```

**Custom Query**:
Use the `--query` argument to provide your own Hebrew sentence.

```bash
python main.py --query "אישה אוכלת תפוח"
```

### Command-Line Arguments

*   `--query`: The Hebrew sentence to search for.
*   `--no-plot`: Disable displaying the results in a plot.
*   `--no-fusion`: Disable query reformulation and fusion ranking.
*   `--force-reindex`: Force a rebuild of the image index. This can be slow.