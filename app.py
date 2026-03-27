import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

from flask import Flask, render_template, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

MAX_INPUT_LENGTH = 5_000  # characters


def load_artifacts() -> Tuple[Any, Any, Optional[str]]:
    """Load the trained model and vectorizer from disk.

    Returns a tuple of (model, vectorizer, error_message).  If loading
    succeeds the error message is None; otherwise both model and
    vectorizer are None.
    """
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        msg = "Model files not found. Run: python model/train_model.py"
        logger.warning(msg)
        return None, None, msg

    try:
        with open(MODEL_PATH, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
    except Exception as exc:
        msg = "Failed to load model files. Re-train the model."
        logger.error("Error loading model artifacts: %s", exc)
        return None, None, msg

    logger.info("Model and vectorizer loaded successfully.")
    return loaded_model, loaded_vectorizer, None


model, vectorizer, load_error = load_artifacts()


@app.route("/")
def home() -> str:
    return render_template("index.html", prediction=None, confidence=None,
                           error=load_error, text="")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    text = request.form.get("news", "").strip()

    if not text:
        return render_template(
            "index.html",
            prediction=None,
            confidence=None,
            error="Please enter some news text.",
            text=text,
        )

    if len(text) > MAX_INPUT_LENGTH:
        return render_template(
            "index.html",
            prediction=None,
            confidence=None,
            error=f"Input is too long. Please limit text to {MAX_INPUT_LENGTH:,} characters.",
            text=text[:MAX_INPUT_LENGTH],
        )

    if load_error:
        return render_template("index.html", prediction=None, confidence=None,
                               error=load_error, text=text)

    try:
        vec = vectorizer.transform([text])
        prediction_raw = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)
        prediction = "Real News" if prediction_raw == 1 else "Fake News"
        logger.info("Prediction: %s (confidence: %.1f%%)", prediction, confidence)
        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            error=None,
            text=text,
        )
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        return render_template(
            "index.html",
            prediction=None,
            confidence=None,
            error="Prediction failed. Re-train model and try again.",
            text=text,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=port)
