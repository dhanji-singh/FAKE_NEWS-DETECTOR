from flask import Flask, render_template, request
import pickle
from pathlib import Path
import os

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, (
            "Model files not found. Run: python model/train_model.py"
        )

    try:
        with open(MODEL_PATH, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
    except Exception:
        return None, None, "Failed to load model files. Re-train the model."

    return loaded_model, loaded_vectorizer, None


model, vectorizer, load_error = load_artifacts()

@app.route('/')
def home():
    return render_template('index.html', prediction=None, error=load_error, text="")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news', '').strip()

    if not text:
        return render_template(
            'index.html',
            prediction=None,
            error='Please enter some news text.',
            text=text
        )

    if load_error:
        return render_template('index.html', prediction=None, error=load_error, text=text)

    try:
        vec = vectorizer.transform([text])
        prediction_raw = int(model.predict(vec)[0])
        prediction = 'Real News' if prediction_raw == 1 else 'Fake News'
        return render_template(
            'index.html',
            prediction=prediction,
            error=None,
            text=text
        )
    except Exception:
        return render_template(
            'index.html',
            prediction=None,
            error='Prediction failed. Re-train model and try again.',
            text=text
        )

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")