#!/usr/bin/env python3
"""
llm_tet_both.py (ENHANCED)

Extended TextTiling pipeline with:
 - Multi-Scale Similarity Fusion (MSF)
 - Adaptive Peak-Picking (PPC)
 - Tiny Logistic Boundary Classifier (TLB) (train or load)
 - Segmentation metrics: Pk, WindowDiff
 - Similarity curve plotting (PNG per doc)

Usage examples (backwards-compatible):
# ST local embeddings
python llm_tet_both.py --model_name all-mpnet-base-v2 --input_dir ./documents --cache_dir ./cache --k 3 --pool mean --threshold 0.40

# MSF + peak picking + plots
python llm_tet_both.py --model_name all-mpnet-base-v2 --input_dir ./documents --cache_dir ./cache --msf --msf_ks 2,3,4 --peak_picking --plot_sims

# Use OpenAI embeddings (set OPENAI_API_KEY first)
python llm_tet_both.py --model_name text-embedding-ada-002 --input_dir ./documents --cache_dir ./cache --embed_mode openai --auto_threshold

# Train tiny classifier (requires labeled JSON for training)
python llm_tet_both.py --model_name all-mpnet-base-v2 --input_dir ./documents --cache_dir ./cache --train_clf train_labels.json --tiny_clf clf.pkl

Outputs:
 - <cache_dir>/predictions_detailed.json
 - per-doc embed caches: <cache_dir>/<docname>_sent_emb.pkl
 - per-doc plots: <cache_dir>/plots/<docname>_sims.png
 - optional saved classifier at --tiny_clf path
"""
import os
import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List, Callable, Optional
import numpy as np
from tqdm.auto import tqdm

import torch
from sentence_transformers import SentenceTransformer, util

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# sklearn / matplotlib used if requested
_SKLEARN_AVAILABLE = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    import joblib
except Exception:
    _SKLEARN_AVAILABLE = False

_PLOTTING_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except Exception:
    _PLOTTING_AVAILABLE = False

def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "<blankline>" in text:
        raw = text.split("<blankline>")
    else:
        text = re.sub(r"\n[ \t\u00A0]+\n", "\n\n", text)
        raw = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in raw if p and p.strip()]
    return paras

_SENT_RE = re.compile(r'(?<=[.!?])\s+')
def split_into_sentences(paragraph: str) -> List[str]:
    para = paragraph.strip().replace("\n", " ")
    sents = [s.strip() for s in _SENT_RE.split(para) if s and s.strip()]
    if not sents:
        parts = [para[i:i+500].strip() for i in range(0, len(para), 500)]
        sents = [p for p in parts if p]
    return sents


def make_st_embedder(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(model_name, device=device)
    def embed(texts: List[str]):
        if not texts:
            return []
        arr = st.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in arr]
    return embed

def make_openai_embedder(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. pip install openai to use OpenAI embeddings.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable to use OpenAI embeddings.")
    openai.api_key = api_key
    def embed(texts: List[str]):
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), 256):
            batch = texts[i:i+256]
            resp = openai.Embedding.create(model=model_name, input=batch)
            out.extend([d["embedding"] for d in resp["data"]])
        return out
    return embed


def embed_sentences_for_doc(doc_sentences: List[str], embed_fn: Callable, cache_path: Path) -> List[List[float]]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return pickle.load(open(cache_path, "rb"))
        except Exception:
            pass
    embeddings = embed_fn(doc_sentences)
    pickle.dump(embeddings, open(cache_path, "wb"))
    return embeddings

def cosine_pool(a_vecs, b_vecs, pool='mean'):
    if len(a_vecs) == 0 or len(b_vecs) == 0:
        return 0.0
    a = torch.tensor(a_vecs)
    b = torch.tensor(b_vecs)
    sim = util.cos_sim(a, b)
    if pool == 'mean':
        return float(sim.mean().item())
    if pool == 'max':
        return float(sim.max().item())
    if pool == 'min':
        return float(sim.min().item())
    raise ValueError("unknown pool")

def compute_sentence_window_sims(sent_embs, k=3, pool="mean"):
    sims = []
    n = len(sent_embs)
    if n < 2:
        return sims
    for i in range(n - 1):
        left = sent_embs[max(0, i - k + 1): i + 1]
        right = sent_embs[i + 1: i + 1 + k]
        sims.append(cosine_pool(left, right, pool))
    return sims

def compute_msf_sims(sent_embs, ks=[2,3,4], pool='mean'):
    if not sent_embs:
        return []
    sims_list = []
    for k in ks:
        s = compute_sentence_window_sims(sent_embs, k=k, pool=pool)
        sims_list.append(np.array(s))
    stacked = np.stack(sims_list, axis=0)
    fused = np.mean(stacked, axis=0)
    return fused.tolist()


def detect_valleys(sims: List[float], prominence: float = 0.03) -> List[int]:
    if not sims:
        return []
    n = len(sims)
    valleys = []
    for i in range(1, n-1):
        if sims[i] < sims[i-1] and sims[i] < sims[i+1]:
            if (max(sims[i-1], sims[i+1]) - sims[i]) >= prominence:
                valleys.append(i)
    return [1 if i in valleys else 0 for i in range(n)]


def build_features_for_sims(sims: List[float]) -> np.ndarray:
    n = len(sims)
    if n == 0:
        return np.zeros((0,6))
    X = []
    for i in range(n):
        s_i = sims[i]
        s_prev = sims[i-1] if i-1 >= 0 else s_i
        s_next = sims[i+1] if i+1 < n else s_i
        left_diff = s_prev - s_i
        right_diff = s_next - s_i
        local_mean = np.mean([s_prev, s_i, s_next])
        X.append([s_i, s_prev, s_next, left_diff, right_diff, local_mean])
    return np.array(X)

def train_tiny_clf_from_labelled_json(train_json_path: Path, out_model_path: Path):
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required to train tiny classifier. pip install scikit-learn joblib")
    data = json.load(open(train_json_path, "r", encoding="utf-8"))
    X_all = []
    y_all = []
    for doc in data:
        sims = doc.get("sims", [])
        y = doc.get("sent_boundaries", [])
        X = build_features_for_sims(sims)
        if X.shape[0] != len(y):
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise RuntimeError("No usable training docs found in train JSON.")
    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_all, y_all)
    joblib.dump(clf, out_model_path)
    return out_model_path

def apply_tiny_clf(clf_path: Path, sims: List[float]) -> List[int]:
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required to use tiny classifier.")
    clf = joblib.load(clf_path)
    X = build_features_for_sims(sims)
    if X.shape[0] == 0:
        return []
    preds = clf.predict_proba(X)[:,1]  

    return [1 if p >= 0.5 else 0 for p in preds]


def predict_sentence_boundaries(sims: List[float], threshold: float) -> List[int]:
    return [1 if s < threshold else 0 for s in sims]

def map_sentence_to_paragraph_boundaries(sent_boundaries: List[int], sentence_to_para: List[int]) -> List[int]:
    para_last_sent = {}
    for s_idx, p_idx in enumerate(sentence_to_para):
        para_last_sent[p_idx] = s_idx
    para_boundaries = set()
    for i, b in enumerate(sent_boundaries):
        if b != 1:
            continue
        p_idx = sentence_to_para[i]
        last_idx = para_last_sent.get(p_idx)
        if last_idx is not None and i == last_idx:
            next_para = p_idx + 1
            para_boundaries.add(next_para)
    return sorted([p+1 for p in para_boundaries])

def pk_metric(reference: List[int], hypothesis: List[int]) -> float:
    n = len(reference) + 1
    if n <= 1:
        return 0.0
    K = max(1, round(n / (sum(reference) + 1) / 2))  
    errors = 0
    denom = n - K
    for i in range(0, denom):
        ref_seg = any(reference[i:i+K])
        hyp_seg = any(hypothesis[i:i+K])
        if ref_seg != hyp_seg:
            errors += 1
    return errors / denom

def window_diff_metric(reference: List[int], hypothesis: List[int]) -> float:
    n = len(reference) + 1
    if n <= 1:
        return 0.0
    B = sum(reference)
    K = max(1, round(n / (B + 1) / 2))
    errors = 0
    denom = n - K
    for i in range(0, denom):
        ref_count = sum(reference[i:i+K])
        hyp_count = sum(hypothesis[i:i+K])
        if ref_count != hyp_count:
            errors += 1
    return errors / denom


def plot_similarity_curve(sims: List[float], sent_boundaries: List[int], out_png: Path, title: Optional[str] = None):
    if not _PLOTTING_AVAILABLE:
        return
    if not sims:
        return
    plt.figure(figsize=(10,4))
    plt.plot(range(len(sims)), sims, marker='o', label='Similarity')
    for i, b in enumerate(sent_boundaries):
        if b == 1:
            plt.axvline(i, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Sentence index (between i and i+1)")
    plt.ylabel("Cosine similarity")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def auto_threshold_from_sims(all_sims: List[float], percentile: float = 40.0) -> float:
    if not all_sims:
        return 0.5
    arr = np.array(all_sims)
    return float(np.percentile(arr, percentile))


def process_documents(input_dir: Path, model_name: str, embed_mode: str, cache_dir: Path,
                      k:int, pool:str, threshold: float, auto_threshold: bool,
                      use_msf: bool, msf_ks: List[int], use_peakpicking: bool, prominence: float,
                      tiny_clf_path: Optional[Path], train_clf_path: Optional[Path],
                      compute_metrics_flag: bool, plot_sims_flag: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cache_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_sims_for_auto = []

    if embed_mode == "openai":
        embed_fn = make_openai_embedder(model_name)
    else:
        embed_fn = make_st_embedder(model_name)

    if train_clf_path:
        print("Training tiny classifier from:", train_clf_path)
        trained_path = train_tiny_clf_from_labelled_json(Path(train_clf_path), tiny_clf_path or (cache_dir / "tiny_clf.pkl"))
        print("Saved trained classifier to:", trained_path)
        tiny_clf_path = Path(trained_path)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = input_dir / fname
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_into_paragraphs(text)

        sentences = []
        sentence_to_para = []
        for p_idx, para in enumerate(paragraphs):
            sents = split_into_sentences(para)
            for s in sents:
                sentences.append(s)
                sentence_to_para.append(p_idx)

        emb_cache = cache_dir / f"{fname}_sent_emb.pkl"
        sent_embs = embed_sentences_for_doc(sentences, embed_fn, emb_cache)

        if use_msf:
            sims = compute_msf_sims(sent_embs, ks=msf_ks, pool=pool)
        else:
            sims = compute_sentence_window_sims(sent_embs, k=k, pool=pool)

        if sims:
            all_sims_for_auto.extend(sims)

        results.append({
            "doc_name": fname,
            "paragraphs": len(paragraphs),
            "n_sentences": len(sentences),
            "sentences": sentences,
            "sentence_to_para": sentence_to_para,
            "sims": sims,
            "sent_boundaries": [],      
            "para_boundaries": []
        })

    if auto_threshold:
        thr = auto_threshold_from_sims(all_sims_for_auto, percentile=40.0)
        print(f"[auto-threshold] chosen = {thr:.4f} (40th percentile of sims)")
    else:
        thr = threshold

    for r in results:
        sims = r["sims"] or []
        if not sims:
            r["sent_boundaries"] = []
            r["para_boundaries"] = []
            continue

        sent_bounds = None

        if tiny_clf_path and tiny_clf_path.exists():
            try:
                sent_bounds = apply_tiny_clf(Path(tiny_clf_path), sims)
            except Exception as e:
                print("Tiny clf apply error:", e)
                sent_bounds = None

        if sent_bounds is None and use_peakpicking:
            sent_bounds = detect_valleys(sims, prominence=prominence)

        if sent_bounds is None:
            sent_bounds = predict_sentence_boundaries(sims, thr)

        max_len = len(r["sentence_to_para"]) - 1
        if len(sent_bounds) > max_len:
            sent_bounds = sent_bounds[:max_len]
        elif len(sent_bounds) < max_len:
            sent_bounds = sent_bounds + [0] * (max_len - len(sent_bounds))

        r["sent_boundaries"] = sent_bounds
        r["para_boundaries"] = map_sentence_to_paragraph_boundaries(sent_bounds, r["sentence_to_para"])

        if plot_sims_flag and _PLOTTING_AVAILABLE:
            out_png = plots_dir / f"{r['doc_name']}_sims.png"
            plot_similarity_curve(sims, sent_bounds, out_png, title=r['doc_name'])

    metrics_summary = {}
    if compute_metrics_flag:
        gold_path = cache_dir / "gold_labels.json"
        if gold_path.exists():
            gold_data = json.load(open(gold_path, "r", encoding="utf-8"))
            for r in results:
                match = next((d for d in gold_data if d.get("doc_name") == r["doc_name"]), None)
                if match:
                    ref = match.get("sent_boundaries", [])
                    hyp = r.get("sent_boundaries", [])
                    if len(ref) != len(hyp):
                        m = min(len(ref), len(hyp))
                        ref = ref[:m]; hyp = hyp[:m]
                    pk = pk_metric(ref, hyp)
                    wd = window_diff_metric(ref, hyp)
                    f1, prec, rec, _ = precision_recall_fscore_support(ref, hyp, average='binary', zero_division=0)
                    metrics_summary[r["doc_name"]] = {"Pk": pk, "WindowDiff": wd, "F1": f1, "Prec": prec, "Recall": rec}
        else:
            print("No gold_labels.json found in cache_dir — skipping compute_metrics (place gold_labels.json in cache_dir to enable).")

    out_path = cache_dir / "predictions_detailed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if metrics_summary:
        with open(cache_dir / "metrics_summary.json", "w", encoding="utf-8") as mf:
            json.dump(metrics_summary, mf, indent=2, ensure_ascii=False)
        print("Saved metrics summary to:", cache_dir / "metrics_summary.json")

    return out_path, thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Embedding model. For OpenAI use text-embedding-.. and set --embed_mode=openai. For local ST models use e.g. all-mpnet-base-v2")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--k", type=int, default=3, help="window size (sentences)")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean","max","min"])
    parser.add_argument("--threshold", type=float, default=0.40, help="manual threshold (cosine similarity cutoff)")
    parser.add_argument("--auto_threshold", action="store_true", help="choose unsupervised threshold using sims percentile")
    parser.add_argument("--embed_mode", type=str, choices=["st","openai"], default="st", help="st = SentenceTransformer, openai = OpenAI embeddings")

    parser.add_argument("--msf", dest="msf", action="store_true", help="Enable Multi-Scale Similarity Fusion")
    parser.add_argument("--msf_ks", type=str, default="2,3,4", help="Comma-separated window sizes for MSF (e.g. 2,3,4)")
    parser.add_argument("--peak_picking", dest="peak_picking", action="store_true", help="Use adaptive peak-picking (valley detection) instead of simple thresholding")
    parser.add_argument("--prominence", type=float, default=0.03, help="Prominence threshold for peak-picking (valley depth)")
    parser.add_argument("--tiny_clf", type=str, default=None, help="Path to tiny classifier pickle to load/apply")
    parser.add_argument("--train_clf", type=str, default=None, help="Path to labelled JSON to train tiny classifier (then saved to --tiny_clf or cache)")
    parser.add_argument("--compute_metrics", dest="compute_metrics", action="store_true", help="Compute Pk and WindowDiff if gold_labels.json present in cache_dir")
    parser.add_argument("--plot_sims", dest="plot_sims", action="store_true", help="Save similarity curve plots to cache_dir/plots")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir)

    msf_ks = [int(x) for x in args.msf_ks.split(",")] if args.msf else []

    tiny_clf_path = Path(args.tiny_clf) if args.tiny_clf else None
    train_clf_path = Path(args.train_clf) if args.train_clf else None

    out_path, chosen_thr = process_documents(
        input_dir,
        args.model_name,
        args.embed_mode,
        cache_dir,
        k=args.k,
        pool=args.pool,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        use_msf=args.msf,
        msf_ks=msf_ks,
        use_peakpicking=args.peak_picking,
        prominence=args.prominence,
        tiny_clf_path=tiny_clf_path,
        train_clf_path=train_clf_path,
        compute_metrics_flag=args.compute_metrics,
        plot_sims_flag=args.plot_sims
    )

    print(f"\nSaved detailed predictions to: {out_path}")
    print(f"Threshold used = {chosen_thr:.4f}")

if __name__ == "__main__":
    main()
