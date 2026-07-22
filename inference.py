import os
import json
import time
import threading

import joblib
import numpy as np
from google.cloud import storage
from sentence_transformers import SentenceTransformer


VALID_LABELS = {
    "ideological",
    "institutionalized",
    "interpersonal",
    "internalized"
}

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")           # e.g. "emp-model-artifacts"
RELOAD_CHECK_INTERVAL_SEC = int(os.environ.get("RELOAD_CHECK_INTERVAL_SEC", "300"))
LOCAL_CACHE_DIR = "/tmp/emp_model_cache"


def build_llm_prompt(sentence: str) -> str:
    return f"""
You are classifying one sentence into exactly one of four oppression labels.

Choose exactly one label from this list:
- ideological
- institutionalized
- interpersonal
- internalized

Definitions:
- ideological: Broad beliefs, stereotypes, cultural assumptions, dominant narratives, or social norms.
- institutionalized: Policies, formal systems, schools, workplaces, organizations, laws, access, or structural barriers.
- interpersonal: Direct treatment between people, such as comments, exclusion, bullying, discrimination, or unfair behavior.
- internalized: Self-directed shame, self-doubt, hiding identity, lowered self-worth, or believing negative ideas about oneself.

Sentence:
\"\"\"{sentence}\"\"\"

Return only valid JSON in this exact format:
{{
  "label": "ideological | institutionalized | interpersonal | internalized",
  "reason": "one short reason"
}}
"""


def classify_with_groq(sentence: str) -> dict:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"label": "llm_unavailable", "reason": "GROQ_API_KEY is not configured."}

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": "You are a strict JSON-only text classification model."},
            {"role": "user", "content": build_llm_prompt(sentence)}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = completion.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"label": "llm_parse_error", "reason": raw}

    label = parsed.get("label", "").strip().lower()
    if label not in VALID_LABELS:
        return {"label": "llm_invalid_label", "reason": str(parsed)}

    return {"label": label, "reason": parsed.get("reason", "")}


class EMPClassifier:
    """
    Loads artifacts (svd.joblib, logreg_pipeline.joblib, metadata.json) from GCS.

    If MODEL_BUCKET is unset, falls back to the artifacts baked into the image
    at Model/logreg_artifacts (useful for local dev / first deploy before the
    closed-loop retraining pipeline exists).

    Periodically re-checks the bucket's "current_version.json" pointer and
    hot-reloads if a newer, promoted model is available -- this is what lets
    the retraining job update the live model without a redeploy.
    """

    def __init__(self, local_artifact_dir="Model/logreg_artifacts"):
        self.local_artifact_dir = local_artifact_dir
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.model_version = "bundled"

        self._load(initial=True)

    def _gcs_client(self):
        return storage.Client()

    def _fetch_current_version(self):
        client = self._gcs_client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob("current_version.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())["version"]

    def _download_version(self, version: str) -> str:
        client = self._gcs_client()
        bucket = client.bucket(MODEL_BUCKET)
        dest_dir = os.path.join(LOCAL_CACHE_DIR, version)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in ("svd.joblib", "logreg_pipeline.joblib", "metadata.json"):
            local_path = os.path.join(dest_dir, fname)
            if not os.path.exists(local_path):
                blob = bucket.blob(f"versions/{version}/{fname}")
                blob.download_to_filename(local_path)

        return dest_dir

    def _load(self, initial=False):
        artifact_dir = self.local_artifact_dir
        version = "bundled"

        if MODEL_BUCKET:
            try:
                remote_version = self._fetch_current_version()
                if remote_version:
                    artifact_dir = self._download_version(remote_version)
                    version = remote_version
            except Exception as e:
                if initial:
                    # No usable remote model yet -- fall back to bundled artifacts.
                    print(f"[EMPClassifier] GCS load failed, using bundled artifacts: {e}")
                else:
                    print(f"[EMPClassifier] reload check failed, keeping current model: {e}")
                    return

        with open(os.path.join(artifact_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        svd = joblib.load(os.path.join(artifact_dir, "svd.joblib"))
        clf = joblib.load(os.path.join(artifact_dir, "logreg_pipeline.joblib"))
        embedder = SentenceTransformer(metadata["embedding_model_name"], device="cpu")

        with self._lock:
            self.metadata = metadata
            self.conf_threshold = metadata["confidence_threshold"]
            self.svd = svd
            self.clf = clf
            self.embedder = embedder
            self.model_version = version
            self._last_check = time.time()

    def _maybe_reload(self):
        if not MODEL_BUCKET:
            return
        if time.time() - self._last_check < RELOAD_CHECK_INTERVAL_SEC:
            return
        try:
            remote_version = self._fetch_current_version()
            if remote_version and remote_version != self.model_version:
                self._load()
            else:
                self._last_check = time.time()
        except Exception as e:
            print(f"[EMPClassifier] reload check failed: {e}")
            self._last_check = time.time()

    def predict_logreg(self, sentence: str) -> dict:
        self._maybe_reload()

        embedding = self.embedder.encode([sentence], normalize_embeddings=True, convert_to_numpy=True)
        X_svd = self.svd.transform(embedding)

        proba = self.clf.predict_proba(X_svd)[0]
        classes = self.clf.named_steps["logreg"].classes_
        sorted_idx = np.argsort(proba)[::-1]

        top1_idx, top2_idx = sorted_idx[0], sorted_idx[1]
        all_probs = {str(label): float(prob) for label, prob in zip(classes, proba)}

        return {
            "top1_label": str(classes[top1_idx]),
            "top1_prob": float(proba[top1_idx]),
            "top2_label": str(classes[top2_idx]),
            "top2_prob": float(proba[top2_idx]),
            "all_probs": all_probs
        }

    def classify(self, sentence: str, use_llm_backup: bool = True) -> dict:
        logreg_result = self.predict_logreg(sentence)

        if logreg_result["top1_prob"] >= self.conf_threshold:
            return {
                "final_label": logreg_result["top1_label"],
                "source": "logreg",
                "llm_used": False,
                "llm_reason": None,
                "logreg": logreg_result
            }

        if not use_llm_backup:
            return {
                "final_label": logreg_result["top1_label"],
                "source": "logreg_low_confidence",
                "llm_used": False,
                "llm_reason": None,
                "logreg": logreg_result
            }

        llm_result = classify_with_groq(sentence)

        if llm_result["label"] in VALID_LABELS:
            final_label = llm_result["label"]
            source = "llm_backup"
            llm_used = True
        else:
            final_label = logreg_result["top1_label"]
            source = "logreg_fallback_after_llm_error"
            llm_used = False

        return {
            "final_label": final_label,
            "source": source,
            "llm_used": llm_used,
            "llm_reason": llm_result.get("reason", ""),
            "logreg": logreg_result
        }