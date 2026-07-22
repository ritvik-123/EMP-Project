import gradio as gr
from inference import EMPClassifier


classifier = EMPClassifier()


def classify_sentence(sentence, use_llm_backup):
    if not sentence or not sentence.strip():
        return {
            "error": "Please enter a sentence."
        }

    result = classifier.classify(
        sentence=sentence,
        use_llm_backup=use_llm_backup
    )

    logreg = result["logreg"]

    return {
        "final_label": result["final_label"],
        "source": result["source"],
        "llm_used": result["llm_used"],
        "llm_reason": result.get("llm_reason"),
        "logreg_top1_label": logreg["top1_label"],
        "logreg_top1_prob": round(logreg["top1_prob"], 4),
        "logreg_top2_label": logreg["top2_label"],
        "logreg_top2_prob": round(logreg["top2_prob"], 4),
        "all_logreg_probs": {
            label: round(prob, 4)
            for label, prob in logreg["all_probs"].items()
        }
    }


demo = gr.Interface(
    fn=classify_sentence,
    inputs=[
        gr.Textbox(
            label="Enter a sentence",
            lines=4,
            placeholder="Example: I started believing I was less capable because of how people treated me."
        ),
        gr.Checkbox(
            label="Use LLM backup for low-confidence predictions",
            value=True
        )
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="EMP Oppression Classifier",
    description=(
        "Classifies a sentence into one of four oppression labels: "
        "ideological, institutionalized, interpersonal, or internalized. "
        "Uses Logistic Regression as the main model and an LLM backup for low-confidence predictions."
    ),
    examples=[
        ["I started believing I was less capable because of how people treated me.", True],
        ["The school policy made it harder for students like me to access support.", True],
        ["People kept making jokes about my background in class.", True],
        ["Society often assumes some groups are naturally less capable.", True],
    ]
)


if __name__ == "__main__":
    demo.launch()