import os
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


VALID_LABELS = {
    "ideological",
    "institutionalized",
    "interpersonal",
    "internalized"
}


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
        return {
            "label": "llm_unavailable",
            "reason": "GROQ_API_KEY is not configured."
        }

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON-only text classification model."
            },
            {
                "role": "user",
                "content": build_llm_prompt(sentence)
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = completion.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "label": "llm_parse_error",
            "reason": raw
        }

    label = parsed.get("label", "").strip().lower()

    if label not in VALID_LABELS:
        return {
            "label": "llm_invalid_label",
            "reason": str(parsed)
        }

    return {
        "label": label,
        "reason": parsed.get("reason", "")
    }


class EMPClassifier:
    def __init__(self, artifact_dir="Model/logreg_artifacts"):
        self.artifact_dir = artifact_dir

        with open(os.path.join(artifact_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.conf_threshold = self.metadata["confidence_threshold"]
        self.embedding_model_name = self.metadata["embedding_model_name"]

        self.svd = joblib.load(os.path.join(artifact_dir, "svd.joblib"))
        self.clf = joblib.load(os.path.join(artifact_dir, "logreg_pipeline.joblib"))

        self.embedder = SentenceTransformer(self.embedding_model_name, device="cpu")

    def predict_logreg(self, sentence: str) -> dict:
        embedding = self.embedder.encode(
            [sentence],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        X_svd = self.svd.transform(embedding)

        proba = self.clf.predict_proba(X_svd)[0]
        classes = self.clf.named_steps["logreg"].classes_

        sorted_idx = np.argsort(proba)[::-1]

        top1_idx = sorted_idx[0]
        top2_idx = sorted_idx[1]

        all_probs = {
            str(label): float(prob)
            for label, prob in zip(classes, proba)
        }

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