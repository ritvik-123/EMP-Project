# EMP — Oppression-Type Sentence Classifier

A sentence-level classifier that categorizes text into one of four oppression types — **ideological**, **institutionalized**, **interpersonal**, or **internalized** — trained on faculty survey data and deployed as a live, self-improving web application on Google Cloud Run.

**Live demo:** https://emp-project-452416352424.us-central1.run.app

---

## Project goal

The core objective was to train a lightweight, deployable **logistic regression classifier** that can take a single sentence and place it along a scale from broad societal belief (*ideological*) down to individual self-perception (*internalized*):

| Label | Meaning |
|---|---|
| **Ideological** | Broad beliefs, stereotypes, cultural assumptions, dominant narratives, or social norms |
| **Institutionalized** | Policies, formal systems, schools, workplaces, organizations, laws, or structural barriers |
| **Interpersonal** | Direct treatment between people — comments, exclusion, bullying, discrimination |
| **Internalized** | Self-directed shame, self-doubt, hiding identity, or lowered self-worth |

Sentences are embedded with `sentence-transformers/all-MiniLM-L6-v2`, reduced via `TruncatedSVD`, and classified with a `LogisticRegression` head — chosen over heavier alternatives for its speed, low deployment footprint, and interpretability.

## Model comparison

Logistic regression wasn't the only architecture tried. As part of model selection, it was benchmarked against **LinearSVC**, an **MLP classifier**, and a fine-tuned **RoBERTa-large (BERT-family)** model, evaluated on the same held-out set of real (non-synthetic) sentences:

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression + SVD (balanced)** | **0.567** | **0.562** |
| LinearSVC + SVD | 0.550 | — |
| Logistic Regression + SVD (unbalanced) | 0.517 | 0.516 |
| Logistic Regression (no SVD) | 0.500 | 0.487 |
| MLP + SVD | 0.383 | 0.390 |
| RoBERTa-large (fine-tuned) | *exploratory — not adopted for production* | |

Logistic regression with SVD dimensionality reduction and class-balanced weighting came out ahead of the other classical baselines on this small real-sentence evaluation set. A fine-tuned RoBERTa-large model was also explored, but given the small dataset size and the cost/latency tradeoff of running a full transformer in production, it wasn't selected as the deployed model — logistic regression offered comparable practical performance at a fraction of the training data requirement, inference cost, and cold-start latency.

**Open question:** these numbers come from a relatively small real-sentence evaluation set, while the production model itself is trained primarily on ~750 LLM-generated synthetic sentences. Synthetic-to-synthetic cross-validation F1 for the production model is closer to ~0.85, which is likely inflated and not fully representative of real-world generalization — this is the exact gap the closed-loop retraining system (below) is designed to close over time.

## Architecture

```
┌─────────────┐      predict      ┌──────────────────┐
│   Browser    │ ─────────────────▶│  Cloud Run       │
│ (index.html) │◀───────────────── │  emp-project     │
└─────────────┘   label + probs    │  (FastAPI)       │
       │                           └─────────┬─────────┘
       │ feedback (correct/incorrect)         │ loads current model
       ▼                                      ▼
┌─────────────┐                     ┌──────────────────┐
│  Firestore   │                     │  GCS bucket      │
│ predictions  │                     │  model artifacts │
│  collection  │                     │  (versioned)     │
└──────┬───────┘                     └────────▲─────────┘
       │ confirmed feedback                    │ promotes new version
       ▼                                       │
┌─────────────────────────────────────────────┴───┐
│  Cloud Run Job: emp-retrain (scheduled)           │
│  synthetic data + real feedback → retrain →       │
│  evaluate on frozen holdout → promote if better   │
└────────────────────────────────────────────────────┘
```

**Inference path:** a sentence is embedded, reduced via SVD, and classified by logistic regression. If the model's top prediction falls below a confidence threshold, it optionally falls back to an LLM (via Groq) for a second opinion.

**Closed-loop retraining:** every prediction is logged to Firestore. When a user confirms whether a prediction was correct (or supplies the correct label), that example is permanently and randomly routed into one of two pools:
- **Train pool (85%)** — folded into the next retraining run, oversampled against the synthetic base data so it isn't drowned out.
- **Eval pool (15%)** — a frozen holdout that is *never* trained on, used to score every retraining candidate on the same fixed yardstick over time.

A scheduled Cloud Run Job retrains periodically, and only promotes a new model (via a version pointer in Cloud Storage) if it beats the currently deployed model's frozen-eval score. The live service hot-reloads the new model within minutes — no redeploy required.

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Dimensionality reduction | `scikit-learn` `TruncatedSVD` |
| Classifier | `scikit-learn` `LogisticRegression` |
| LLM fallback | Groq API |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Feedback storage | Firestore (Native mode) |
| Model artifact storage | Google Cloud Storage |
| Hosting | Cloud Run (service) |
| Retraining | Cloud Run (job) + Cloud Scheduler |
| Container builds | Cloud Build |

## Repository structure

```
├── main.py                  # FastAPI app: /, /predict, /feedback, /health
├── inference.py              # EMPClassifier: embedding, SVD, logreg, GCS hot-reload
├── requirements.txt
├── Dockerfile                 # web service image
├── templates/index.html       # frontend (dynamic, feedback UI, dark theme)
├── retrain/
│   ├── retrain.py             # closed-loop retraining job
│   ├── Dockerfile             # retraining job image
│   ├── requirements.txt
│   └── Data/                  # synthetic training data (copy)
├── Model/logreg_artifacts/    # bundled fallback model artifacts
├── Data/                      # synthetic + real training/eval CSVs
└── Current-Notebooks/         # model development & comparison notebooks
```

## Running locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key          # optional, for low-confidence fallback
uvicorn main:app --reload
```

Without `MODEL_BUCKET` set, the app falls back to the bundled artifacts in `Model/logreg_artifacts/`. Firestore logging requires `GOOGLE_APPLICATION_CREDENTIALS` or running within an authenticated `gcloud` environment.

## Deployment

Deployed on Google Cloud Run. See commit history/project notes for the full setup (Firestore, GCS bucket, IAM bindings, Cloud Scheduler). Core commands:

```bash
# Web service
gcloud builds submit --tag gcr.io/PROJECT_ID/emp-project
gcloud run deploy emp-project --image gcr.io/PROJECT_ID/emp-project --region us-central1

# Retraining job (from within retrain/)
gcloud builds submit --tag gcr.io/PROJECT_ID/emp-retrain
gcloud run jobs update emp-retrain --image gcr.io/PROJECT_ID/emp-retrain --region us-central1
```
