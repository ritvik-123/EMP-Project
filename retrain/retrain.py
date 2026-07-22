"""
Closed-loop retraining job.

1. Pulls confirmed feedback from Firestore (`predictions` collection).
   - pool == "train": usable for training (both "correct" and "incorrect"
     feedback count -- the ground truth is predicted_label in the first
     case, corrected_label in the second).
   - pool == "eval": NEVER trains. This is the frozen holdout used to
     score every retrain, so improvement numbers are comparable over time.
2. Combines the real "train" pool with the original synthetic dataset,
   oversampling the real examples so they aren't drowned out while small.
3. Retrains embedder-frozen / SVD / logistic-regression pipeline, matching
   the existing architecture (all-MiniLM-L6-v2 -> TruncatedSVD(50) -> LogReg).
4. Evaluates the new pipeline against the frozen eval pool, and only
   promotes it (uploads to GCS, flips the current_version pointer) if it's
   at least as good as the currently deployed model AND the eval pool is
   large enough to trust the number.
"""

import os
import io
import json
import glob
import time
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from google.cloud import firestore, storage


MODEL_BUCKET = os.environ["MODEL_BUCKET"]
SYNTHETIC_DATA_GLOB = os.environ.get("SYNTHETIC_DATA_GLOB", "Data/Generated_Sentences_1*.csv")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_SVD_COMPONENTS = 50
CONFIDENCE_THRESHOLD = 0.75
RANDOM_STATE = 42
REAL_DATA_OVERSAMPLE = int(os.environ.get("REAL_DATA_OVERSAMPLE", "3"))

MIN_EVAL_EXAMPLES = int(os.environ.get("MIN_EVAL_EXAMPLES", "20"))
MIN_NEW_TRAIN_EXAMPLES = int(os.environ.get("MIN_NEW_TRAIN_EXAMPLES", "10"))
F1_REGRESSION_TOLERANCE = float(os.environ.get("F1_REGRESSION_TOLERANCE", "0.0"))

LABELS = ["ideological", "institutionalized", "interpersonal", "internalized"]


def load_synthetic_data() -> pd.DataFrame:
    frames = []
    for path in glob.glob(SYNTHETIC_DATA_GLOB):
        df = pd.read_csv(path)
        df = df.rename(columns={"Sentence": "sentence", "Label": "label"})
        frames.append(df[["sentence", "label"]])
    if not frames:
        raise RuntimeError(f"No synthetic data found matching {SYNTHETIC_DATA_GLOB}")
    return pd.concat(frames, ignore_index=True)


def load_feedback_pools(db: firestore.Client):
    docs = db.collection("predictions").where("feedback", "!=", None).stream()

    train_rows, eval_rows = [], []
    for doc in docs:
        d = doc.to_dict()
        label = d["corrected_label"] if d["feedback"] == "incorrect" else d["predicted_label"]
        if label not in LABELS:
            continue
        row = {"sentence": d["sentence"], "label": label}
        if d.get("pool") == "eval":
            eval_rows.append(row)
        elif d.get("pool") == "train":
            train_rows.append(row)

    return pd.DataFrame(train_rows), pd.DataFrame(eval_rows)


def get_retrain_state(db: firestore.Client) -> dict:
    doc = db.collection("retrain_state").document("status").get()
    if doc.exists:
        return doc.to_dict()
    return {"current_model_eval_f1": None, "last_train_pool_size": 0, "current_version": None}


def fit_pipeline(embedder, sentences, labels):
    embeddings = embedder.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)

    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(embeddings)

    clf = Pipeline([
        ("logreg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    clf.fit(X_svd, labels)

    return svd, clf


def evaluate(embedder, svd, clf, eval_df: pd.DataFrame) -> float:
    embeddings = embedder.encode(eval_df["sentence"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    X_svd = svd.transform(embeddings)
    preds = clf.predict(X_svd)
    return f1_score(eval_df["label"], preds, average="macro")


def upload_version(bucket, version: str, svd, clf, metadata: dict):
    tmp_dir = f"/tmp/{version}"
    os.makedirs(tmp_dir, exist_ok=True)

    joblib.dump(svd, os.path.join(tmp_dir, "svd.joblib"))
    joblib.dump(clf, os.path.join(tmp_dir, "logreg_pipeline.joblib"))
    with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    for fname in ("svd.joblib", "logreg_pipeline.joblib", "metadata.json"):
        bucket.blob(f"versions/{version}/{fname}").upload_from_filename(os.path.join(tmp_dir, fname))

    bucket.blob("current_version.json").upload_from_string(json.dumps({"version": version}))


def main():
    db = firestore.Client()
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)

    state = get_retrain_state(db)
    train_df, eval_df = load_feedback_pools(db)

    print(f"Real train pool: {len(train_df)} examples | Real eval pool: {len(eval_df)} examples")

    new_examples_since_last = len(train_df) - state.get("last_train_pool_size", 0)
    if new_examples_since_last < MIN_NEW_TRAIN_EXAMPLES:
        print(f"Only {new_examples_since_last} new confirmed examples since last retrain "
              f"(threshold {MIN_NEW_TRAIN_EXAMPLES}). Skipping.")
        return

    synthetic_df = load_synthetic_data()

    # Oversample real data so it isn't drowned out by ~750 synthetic rows.
    real_oversampled = pd.concat([train_df] * REAL_DATA_OVERSAMPLE, ignore_index=True) if len(train_df) else train_df
    combined_df = pd.concat([synthetic_df, real_oversampled], ignore_index=True)

    print(f"Training on {len(synthetic_df)} synthetic + "
          f"{len(real_oversampled)} (oversampled x{REAL_DATA_OVERSAMPLE}) real = {len(combined_df)} rows")

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    svd, clf = fit_pipeline(embedder, combined_df["sentence"].tolist(), combined_df["label"].tolist())

    if len(eval_df) < MIN_EVAL_EXAMPLES:
        print(f"Frozen eval pool only has {len(eval_df)} examples "
              f"(need {MIN_EVAL_EXAMPLES}) -- can't reliably score this candidate. "
              f"Skipping promotion; will retry once the eval pool has grown.")
        return

    new_f1 = evaluate(embedder, svd, clf, eval_df)
    current_f1 = state.get("current_model_eval_f1")

    print(f"Candidate model eval F1 (macro, frozen holdout): {new_f1:.4f} "
          f"| Currently deployed: {current_f1}")

    if current_f1 is not None and new_f1 < current_f1 - F1_REGRESSION_TOLERANCE:
        print("Candidate is worse than the deployed model. Not promoting.")
        return

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "n_svd_components": N_SVD_COMPONENTS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "labels": sorted(LABELS),
        "random_state": RANDOM_STATE,
        "trained_at": version,
        "real_train_examples": len(train_df),
        "eval_f1_macro": new_f1,
    }

    upload_version(bucket, version, svd, clf, metadata)

    db.collection("retrain_state").document("status").set({
        "current_model_eval_f1": new_f1,
        "last_train_pool_size": len(train_df),
        "current_version": version,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    print(f"Promoted new model version {version} (eval F1 {new_f1:.4f}). "
          f"Live service will pick it up on its next reload check.")


if __name__ == "__main__":
    main()