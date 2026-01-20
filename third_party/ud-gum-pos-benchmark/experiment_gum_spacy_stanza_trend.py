#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiment_gum_spacy_stanza_trend_with_heatmap.py

spaCy vs Stanza on UD English GUM:
- POS Accuracy trend (aligned tokens)
- Tokenization F1 trend (span-based)
- Speed trend (sent/sec)
- POS confusion heatmaps (aligned tokens, row-normalized)
  * generated on the final evaluation size (default 500)

Outputs (PDF):
- pos_accuracy_trend.pdf
- token_f1_trend.pdf
- speed_trend.pdf
- spacy_pos_confusion.pdf
- stanza_pos_confusion.pdf

Usage:
python experiment_gum_spacy_stanza_trend_with_heatmap.py --conllu Data/en_gum-ud-dev.conllu
"""

import argparse
import time
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from conllu import parse_incr
import spacy
import stanza


# =====================
# Load CoNLL-U
# =====================
def load_conllu_sentences(path: str, max_sentences: int) -> List[Dict]:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            tokens, pos = [], []
            for tok in sent:
                if isinstance(tok.get("id"), int):  # ignore multi-word token rows (e.g., 1-2)
                    form = tok.get("form")
                    upos = tok.get("upos") or "X"
                    if form is None:
                        continue
                    tokens.append(form)
                    pos.append(upos)
            if tokens:
                sentences.append({
                    "text": " ".join(tokens),
                    "gold_tokens": tokens,
                    "gold_pos": pos
                })
            if len(sentences) >= max_sentences:
                break
    return sentences


# =====================
# Tokenization F1 (span-based)
# =====================
def token_spans(tokens: List[str], text: str) -> set:
    spans = []
    start = 0
    for tok in tokens:
        idx = text.find(tok, start)
        if idx == -1:
            idx = text.find(tok)  # fallback
            if idx == -1:
                continue
        spans.append((idx, idx + len(tok)))
        start = idx + len(tok)
    return set(spans)


def token_f1(gold_tokens: List[str], pred_tokens: List[str], text: str) -> float:
    gold = token_spans(gold_tokens, text)
    pred = token_spans(pred_tokens, text)
    tp = len(gold & pred)
    # F1 = 2TP / (|gold| + |pred|)
    return (2 * tp) / (len(gold) + len(pred) + 1e-9)


# =====================
# POS Accuracy (aligned tokens only)
# =====================
def pos_acc_aligned(gold_tok, gold_pos, pred_tok, pred_pos) -> float:
    correct, total = 0, 0
    for gt, gp, pt, pp in zip(gold_tok, gold_pos, pred_tok, pred_pos):
        if gt == pt:
            total += 1
            if gp == pp:
                correct += 1
    return correct / total if total else 0.0


# =====================
# spaCy / Stanza inference
# =====================
def run_spacy(nlp, sentences: List[Dict]) -> List[Dict]:
    out = []
    for s in sentences:
        doc = nlp(s["text"])
        out.append({"tokens": [t.text for t in doc], "pos": [t.pos_ for t in doc]})
    return out


def run_stanza(nlp, sentences: List[Dict]) -> List[Dict]:
    out = []
    for s in sentences:
        doc = nlp(s["text"])
        tokens, pos = [], []
        for sent in doc.sentences:
            for w in sent.words:
                tokens.append(w.text)
                pos.append(w.upos)
        out.append({"tokens": tokens, "pos": pos})
    return out


# =====================
# Speed
# =====================
def measure_speed(run_func, nlp, sentences: List[Dict]) -> float:
    t0 = time.time()
    run_func(nlp, sentences)
    dt = time.time() - t0
    return len(sentences) / dt if dt > 0 else 0.0


# =====================
# Confusion Matrix Heatmap (aligned)
# =====================
UD_UPOS = [
    "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM",
    "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
]

def build_pos_confusion(sentences: List[Dict], tool_output: List[Dict], labels=UD_UPOS) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for s, out in zip(sentences, tool_output):
        for gt, gp, pt, pp in zip(
            s["gold_tokens"], s["gold_pos"],
            out["tokens"], out["pos"]
        ):
            if gt != pt:
                continue  # aligned tokens only
            gp = gp if gp in idx else "X"
            pp = pp if pp in idx else "X"
            cm[idx[gp], idx[pp]] += 1

    return cm


def plot_heatmap_row_normalized(cm: np.ndarray, labels: List[str], title: str, out_pdf: str):
    cm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, aspect="auto")  # default colormap
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted UPOS")
    plt.ylabel("Gold UPOS")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.show()
    print(f"Saved {out_pdf}")


# =====================
# Line plot helper
# =====================
def plot_line(x, y1, y2, title, ylabel, out_pdf):
    plt.figure()
    plt.plot(x, y1, marker="o", label="spaCy")
    plt.plot(x, y2, marker="o", label="Stanza")
    plt.xlabel("Number of Evaluation Sentences")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.show()
    print(f"Saved {out_pdf}")


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conllu", required=True, help="Path to UD GUM .conllu file")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy model name")
    args = parser.parse_args()

    # Evaluation sizes for trend
    SIZES = [100, 200, 300, 400, 500]

    print("=== Loading pipelines ===")
    nlp_spacy = spacy.load(args.spacy_model)
    nlp_stanza = stanza.Pipeline(lang="en", processors="tokenize,pos")

    spacy_pos, stanza_pos = [], []
    spacy_f1s, stanza_f1s = [], []
    spacy_sps, stanza_sps = [], []

    # keep final outputs for heatmap (use last size)
    final_sentences = None
    final_spacy_out = None
    final_stanza_out = None
    final_n = None

    for n in SIZES:
        print(f"\n=== Evaluating with {n} sentences ===")
        sentences = load_conllu_sentences(args.conllu, max_sentences=n)

        # inference outputs
        sp_out = run_spacy(nlp_spacy, sentences)
        st_out = run_stanza(nlp_stanza, sentences)

        # speed (measured independently for this size)
        sp_sps = measure_speed(run_spacy, nlp_spacy, sentences)
        st_sps = measure_speed(run_stanza, nlp_stanza, sentences)

        # metrics (avg over sentences)
        sp_pos = float(np.mean([
            pos_acc_aligned(s["gold_tokens"], s["gold_pos"], o["tokens"], o["pos"])
            for s, o in zip(sentences, sp_out)
        ]))
        st_pos = float(np.mean([
            pos_acc_aligned(s["gold_tokens"], s["gold_pos"], o["tokens"], o["pos"])
            for s, o in zip(sentences, st_out)
        ]))

        sp_f1 = float(np.mean([
            token_f1(s["gold_tokens"], o["tokens"], s["text"])
            for s, o in zip(sentences, sp_out)
        ]))
        st_f1 = float(np.mean([
            token_f1(s["gold_tokens"], o["tokens"], s["text"])
            for s, o in zip(sentences, st_out)
        ]))

        spacy_pos.append(sp_pos)
        stanza_pos.append(st_pos)
        spacy_f1s.append(sp_f1)
        stanza_f1s.append(st_f1)
        spacy_sps.append(sp_sps)
        stanza_sps.append(st_sps)

        # guaranteed console output
        print(f"spaCy  POS Acc(aligned): {sp_pos:.4f} | Tok F1: {sp_f1:.4f} | Speed: {sp_sps:.2f} sent/s")
        print(f"Stanza POS Acc(aligned): {st_pos:.4f} | Tok F1: {st_f1:.4f} | Speed: {st_sps:.2f} sent/s")

        # store last run outputs for heatmap
        if n == SIZES[-1]:
            final_n = n
            final_sentences = sentences
            final_spacy_out = sp_out
            final_stanza_out = st_out

    # ---- line plots ----
    plot_line(SIZES, spacy_pos, stanza_pos,
              "POS Accuracy vs Evaluation Size (UD English GUM)",
              "POS Accuracy (aligned tokens)",
              "pos_accuracy_trend.pdf")

    plot_line(SIZES, spacy_f1s, stanza_f1s,
              "Tokenization F1 vs Evaluation Size (UD English GUM)",
              "Tokenization F1 (span-based)",
              "token_f1_trend.pdf")

    plot_line(SIZES, spacy_sps, stanza_sps,
              "Speed vs Evaluation Size (UD English GUM)",
              "Throughput (sentences/sec)",
              "speed_trend.pdf")

    # ---- heatmaps on final size ----
    print(f"\n=== Generating POS confusion heatmaps on the final run (n={final_n}) ===")

    cm_spacy = build_pos_confusion(final_sentences, final_spacy_out, labels=UD_UPOS)
    cm_stanza = build_pos_confusion(final_sentences, final_stanza_out, labels=UD_UPOS)

    plot_heatmap_row_normalized(
        cm_spacy, UD_UPOS,
        f"spaCy POS Confusion (Aligned Tokens, Row-Normalized, n={final_n})",
        "spacy_pos_confusion.pdf"
    )

    plot_heatmap_row_normalized(
        cm_stanza, UD_UPOS,
        f"Stanza POS Confusion (Aligned Tokens, Row-Normalized, n={final_n})",
        "stanza_pos_confusion.pdf"
    )


if __name__ == "__main__":
    main()
