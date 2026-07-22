from functools import lru_cache

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from inference import EMPClassifier


app = FastAPI(title="EMP Oppression Classifier")

templates = Jinja2Templates(directory="templates")


@lru_cache(maxsize=1)
def get_classifier():
    return EMPClassifier()


class PredictionRequest(BaseModel):
    sentence: str
    use_llm_backup: bool = True


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "sentence": "",
            "use_llm_backup": True
        }
    )


@app.post("/", response_class=HTMLResponse)
def classify_from_form(
    request: Request,
    sentence: str = Form(...),
    use_llm_backup: bool = Form(False)
):
    classifier = get_classifier()

    result = classifier.classify(
        sentence=sentence,
        use_llm_backup=use_llm_backup
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "sentence": sentence,
            "use_llm_backup": use_llm_backup
        }
    )


@app.post("/predict")
def predict(payload: PredictionRequest):
    classifier = get_classifier()

    result = classifier.classify(
        sentence=payload.sentence,
        use_llm_backup=payload.use_llm_backup
    )

    return result


@app.get("/health")
def health():
    return {"status": "ok"}