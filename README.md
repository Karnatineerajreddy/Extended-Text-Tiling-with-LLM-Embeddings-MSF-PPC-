Extended TextTiling with LLM Embeddings (MSF + Peak Picking)
==================================================================================================================================
This repository implements an enhanced TextTiling algorithm for semantic topic segmentation using Large Language Model (LLM) embeddings.
The system uses SentenceTransformer embeddings, Multi-Scale Similarity Fusion (MSF), and Adaptive Peak-Picking to detect topic shifts across long documents.

The output includes:

1. Sentence boundaries

2. Paragraph boundaries

3. Similarity curves

4. JSON + HTML visualization

<p align="center"> <img src="assets/system_diagram.png" width="550" alt="System Diagram"> </p>

==================================================================================================================================
How to Run
1. Run the main segmentation pipeline
python main.py --model_name all-mpnet-base-v2 \
    --input_dir ./documents \
    --cache_dir ./cache \
    --msf --msf_ks 2,3,4 \
    --peak_picking \
    --plot_sims

2. (Optional) Use OpenAI embeddings
python main.py --model_name text-embedding-ada-002 \
    --embed_mode openai \
    --input_dir ./documents \
    --cache_dir ./cache \
    --auto_threshold

3. Generate HTML visualizations
python render_output.py \
    --predictions cache/predictions_detailed.json \
    --out_dir html_output

==================================================================================================================================

Features

1. LLM-Based Embeddings: Supports SentenceTransformers and OpenAI embedding models.

2. Multi-Scale Similarity Fusion (MSF) :Averages similarity curves from multiple window sizes for more stable segmentation.

3. Adaptive Peak-Picking: Automatically detects strong valleys in the similarity curve indicating topic shifts.

4. Optional Tiny Logistic Classifier: When labeled data is available, a lightweight classifier predicts boundaries using similarity-based features.

Rich Outputs

1. JSON with all segmentation metadata

2. PNG similarity plots

3. HTML segmented text for demo purposes

==================================================================================================================================

Algorithm Workflow

1. Text Preprocessing: Split text into paragraphs and sentences using regex-based splitting.

2. Sentence Embedding Extraction: Embed each sentence using SentenceTransformer or OpenAI.

3. Similarity Curve Construction: Compute left/right window similarity and perform MSF.

4. Boundary Detection

5. Thresholding

6. Peak-picking

7. Optional classifier

8. Output Generation: Produce JSON, PNG plots, and HTML segment views.

<p align="center"> <img src="assets/msf_diagram.png" width="500" alt="MSF Diagram"> </p>

==================================================================================================================================

Supported Embedding Models

SentenceTransformer (local)

1. all-mpnet-base-v2 (recommended)

2. all-MiniLM-L6-v2

OpenAI (API required)

1. text-embedding-ada-002

2. text-embedding-3-small

3. text-embedding-3-large

==================================================================================================================================

Dataset

The project uses a curated dataset (Dataset C) including:

1. Analytical articles

2. Fictional narrative passages

3. Multi-section essays


==================================================================================================================================

Installation

git clone https://github.com/yourusername/Extended-TextTiling-LLM.git
cd Extended-TextTiling-LLM
pip install -r requirements.txt


If using OpenAI embeddings:

export OPENAI_API_KEY="your-api-key"
==================================================================================================================================

Example Output
JSON output

Located at:

cache/predictions_detailed.json

Similarity plots
cache/plots/<filename>_sims.png

HTML segmented view
html_output/<filename>.html
==================================================================================================================================

Future Improvements

1. Transformer-based end-to-end segmentation

2. Improved sentence splitting (spaCy/Stanza)

3. Larger annotated dataset for classifier training

4. Multilingual support

5. Interactive visual analytics
==================================================================================================================================

Authors

1. Neeraj Reddy Karnati – System integration, MSF design

2. Abhinav Reddy Telakala – Peak-picking + visualization

3. Pranav M.R. Madduri – Embedding pipeline + classifier