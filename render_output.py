#!/usr/bin/env python3
"""
render_output.py

Reads predictions_detailed.json from your text-tiling pipeline,
and produces beautiful HTML pages visualizing the segmentation.

Usage:
python render_output.py --predictions cache/predictions_detailed.json --out_dir html_output
"""

import json
import argparse
from pathlib import Path

# -------------------------------------------------------
# HTML Template
# -------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{doc_name} — Topic Segmentation</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f0f2f5;
}}

.segment {{
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 22px;
    border-left: 8px solid #4a90e2;
}}

.segment:nth-child(even) {{
    background: #e8f4fc;
    border-left-color: #4a90e2;
}}

.segment:nth-child(odd) {{
    background: #fff3e6;
    border-left-color: #e67e22;
}}

h2 {{
    margin-bottom: 12px;
}}

p {{
    font-size: 1.15em;
    line-height: 1.6em;
}}
</style>
</head>

<body>

<h1>Document: {doc_name}</h1>
<h3>Total Sentences: {n_sentences}</h3>
<h3>Detected Segments: {n_segments}</h3>

<hr><br>

{segments_html}

</body>
</html>
"""

# -------------------------------------------------------
# Segment grouping utility
# -------------------------------------------------------
def build_segments(sentences, sent_bounds):
    segments = []
    current = []

    for i, s in enumerate(sentences):
        current.append(s)

        # boundary => end of segment
        if i < len(sent_bounds) and sent_bounds[i] == 1:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    return segments


# -------------------------------------------------------
# Convert segments → HTML
# -------------------------------------------------------
def segments_to_html(segments):
    html = []
    for idx, seg in enumerate(segments, start=1):
        block = f"<div class='segment'><h2>Segment {idx}</h2>"
        for s in seg:
            block += f"<p>{s}</p>"
        block += "</div>"
        html.append(block)
    return "\n".join(html)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions_detailed.json")
    parser.add_argument("--out_dir", default="html_output", help="Directory to save HTML files")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    data = json.load(open(pred_path, "r", encoding="utf-8"))

    for doc in data:
        doc_name = doc["doc_name"]
        sentences = doc["sentences"]
        sent_bounds = doc["sent_boundaries"]

        segments = build_segments(sentences, sent_bounds)
        segments_html = segments_to_html(segments)

        html_output = HTML_TEMPLATE.format(
            doc_name=doc_name,
            n_sentences=len(sentences),
            n_segments=len(segments),
            segments_html=segments_html,
        )

        out_file = out_dir / f"{doc_name}.html"
        out_file.write_text(html_output, encoding="utf-8")

        print(f"✔ Created HTML for {doc_name} → {out_file}")

    print("\n🎉 All HTML visualizations generated!")


if __name__ == "__main__":
    main()
