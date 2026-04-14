"""
Re-embed NPTEL course catalog using the Gemini Embedding API.

Replaces SentenceTransformer (BAAI/bge-base-en-v1.5, ~440 MB RAM) with
Gemini's cloud embedding model (zero local RAM after startup).

Model  : models/gemini-embedding-2-preview (768-d Matryoshka)
Input  : Data/nptel_courses_with_embeddings.xlsx  (or Data/nptel_courses.csv)
Output : Data/nptel_courses_with_embeddings.xlsx  (same file, embedding column replaced)

Usage:
    # From ScholarGen/ root:
    python scripts/generate_embeddings.py

    # Resume from a checkpoint after an interruption:
    python scripts/generate_embeddings.py --resume
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

# ── Configuration ────────────────────────────────────────────────────────────

API_KEYS = [
    "AIzaSyA95_0rUP-D0lMs_fev8TM8PffzYUZMazI",
    "AIzaSyCJ430RfUl-_JYVqFTo4Fy1_IVRcQv0UyE",
    "AIzaSyDbuDMLZT6tRsEp8lxITRMC5zjWx02LZC0",
    "AIzaSyBioOyJe-Ti4mtnRPu9tOwlSmXbXm7k5g8",
]

MODEL = "models/gemini-embedding-2-preview"
TASK_TYPE = "RETRIEVAL_DOCUMENT"
OUTPUT_DIM = 768          # Matryoshka — 768 is the full dimension

DELAY_PER_REQUEST = 2.16  # seconds (~28 RPM, ~3× conservative — ~117 min total)
CHECKPOINT_EVERY  = 50    # save progress every N rows
BATCH_KEY_SWITCH  = 1     # rotate key every N requests (1 = per request)

# ── Paths ────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_HERE)

EXCEL_IN   = os.path.join(ROOT_DIR, "Data", "nptel_courses_with_embeddings1.xlsx")  # renamed backup
EXCEL_OUT  = os.path.join(ROOT_DIR, "Data", "nptel_courses_with_embeddings.xlsx")
CKPT_FILE  = os.path.join(ROOT_DIR, "Data", "embed_checkpoint.json")
VECTORS_OUT = os.path.join(ROOT_DIR, "Data", "nptel_vectors_768d.npy")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint() -> int:
    """Return the index to resume from (0 if no checkpoint)."""
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            data = json.load(f)
        idx = data.get("next_index", 0)
        print(f"[checkpoint] Resuming from row {idx}")
        return idx
    return 0


def save_checkpoint(next_index: int, embeddings_so_far: list[list[float]]):
    """Persist progress so we can resume after any interruption."""
    with open(CKPT_FILE, "w") as f:
        json.dump({"next_index": next_index}, f)
    # Also dump partial vectors to disk so nothing is lost
    if embeddings_so_far:
        np.save(VECTORS_OUT + ".partial.npy", np.array(embeddings_so_far, dtype=np.float32))


def embed_text(text: str, key_index: int) -> list[float]:
    """
    Call Gemini Embedding API and return the embedding vector.
    Rotates across API_KEYS on each call to spread quota usage.
    """
    key = API_KEYS[key_index % len(API_KEYS)]
    client = genai.Client(api_key=key)
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=TASK_TYPE,
            output_dimensionality=OUTPUT_DIM,
        ),
    )
    return result.embeddings[0].values


def build_text(row: pd.Series) -> str:
    """
    Return the pre-built description field, which already contains:
    course_name, Discipline, Professors, Institute, URL.
    Matches exactly what the original SentenceTransformer embeddings used.
    """
    return str(row.get("description", ""))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume: bool = False):
    # Validate keys
    for i, k in enumerate(API_KEYS):
        if not k or len(k) < 20:
            print(f"ERROR: API_KEYS[{i}] looks invalid. Check the script.")
            sys.exit(1)

    # Load data
    print(f"Loading course data from: {EXCEL_IN}")
    df = pd.read_excel(EXCEL_IN)
    print(f"Total courses: {len(df)}")

    start_idx = load_checkpoint() if resume else 0

    # Load partial embeddings if resuming
    if resume and os.path.exists(VECTORS_OUT + ".partial.npy"):
        partial = np.load(VECTORS_OUT + ".partial.npy").tolist()
        embeddings: list[list[float]] = partial
        print(f"[resume] Loaded {len(embeddings)} existing embeddings")
    else:
        embeddings = [None] * start_idx  # placeholders for already-processed rows

    # Embed remaining rows
    total = len(df)
    errors = 0

    for i in range(start_idx, total):
        row = df.iloc[i]
        text = build_text(row)

        try:
            vec = embed_text(text, key_index=i)
            embeddings.append(vec)
        except Exception as e:
            print(f"  [row {i}] ERROR: {e}  — using zero vector as fallback")
            embeddings.append([0.0] * OUTPUT_DIM)
            errors += 1

        # Progress
        done = i - start_idx + 1
        total_remaining = total - start_idx
        pct = 100 * done / total_remaining
        key_used = i % len(API_KEYS)
        print(f"  [{i+1}/{total}] {pct:.1f}%  key={key_used}  \"{text[:60]}\"")

        # Rate limit
        time.sleep(DELAY_PER_REQUEST)

        # Checkpoint
        if done % CHECKPOINT_EVERY == 0:
            save_checkpoint(i + 1, embeddings)
            print(f"  [checkpoint] Saved at row {i+1}")

    # Fill in any None placeholders (rows before resume point already had embeddings)
    if resume and start_idx > 0:
        import ast
        print("Re-loading pre-resume embeddings from original Excel...")
        df_orig = pd.read_excel(EXCEL_IN)
        for j in range(start_idx):
            try:
                embeddings[j] = ast.literal_eval(str(df_orig["embedding"].iloc[j]))
            except Exception:
                embeddings[j] = [0.0] * OUTPUT_DIM

    # Save results
    print("\nSaving results...")
    df["embedding"] = [json.dumps(e) for e in embeddings]
    df.to_excel(EXCEL_OUT, index=False)
    print(f"Saved Excel: {EXCEL_OUT}")

    np.save(VECTORS_OUT, np.array(embeddings, dtype=np.float32))
    print(f"Saved vectors: {VECTORS_OUT}  shape={np.array(embeddings).shape}")

    # Clean up checkpoint
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    partial_path = VECTORS_OUT + ".partial.npy"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"\nDone! {total} courses embedded, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)
