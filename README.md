# 🛡️ Cyber Kavach – Fake News Detector

A machine-learning web application that classifies news text as **Real** or **Fake**, complete with a confidence score.  
Built with Flask, scikit-learn TF-IDF + Logistic Regression, and a dark-themed UI.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Training the Model](#training-the-model)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)

---

## Demo

1. Paste a news headline or full article into the text box.
2. Click **Analyze**.
3. The app returns a verdict (**Real News** / **Fake News**) along with a confidence percentage.

---

## Features

- 📰 Binary classification: **Real News** vs **Fake News**
- 📊 **Confidence score** shown with each prediction
- ⚡ Fast TF-IDF + Logistic Regression inference (no GPU required)
- 🌑 Dark-themed responsive UI
- 🛡️ Input validation and graceful error handling
- 📝 Structured logging for easy debugging

---

## Architecture

```
User browser
     │  POST /predict (news text)
     ▼
Flask app (app.py)
     │  vectorizer.transform(text)
     ▼
TF-IDF Vectorizer  ──▶  Logistic Regression model
                              │
                              ▼
                    "Real News" / "Fake News" + confidence %
```

**Training pipeline** (`model/train_model.py`):

```
fake_news.csv  ──┐
                 ├──▶ concat + label ──▶ TF-IDF fit ──▶ LogReg fit ──▶ .pkl artifacts
true_news.csv  ──┘
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.9
- `pip`

### 1 – Clone & install

```bash
git clone https://github.com/dhanji-singh/FAKE_NEWS-DETECTOR.git
cd FAKE_NEWS-DETECTOR
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 – Train the model

Place your `fake_news.csv` and `true_news.csv` datasets inside the `model/` directory  
(both must contain at least a `text` column), then run:

```bash
python model/train_model.py
```

This produces `model/fake_model.pkl` and `model/vectorizer.pkl`.

### 3 – Start the web app

```bash
python app.py
```

Open <http://localhost:5000> in your browser.

---

## Training the Model

| File | Expected columns |
|------|-----------------|
| `model/fake_news.csv` | `text` (and optionally `title`) |
| `model/true_news.csv` | `text` (and optionally `title`) |

The script prints train/test accuracy and a classification report to stdout after training.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
FAKE_NEWS-DETECTOR/
├── app.py                  # Flask web application
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # Tool configuration (pytest, flake8)
├── model/
│   ├── train_model.py      # Training script
│   ├── fake_news.csv       # (local only – not committed)
│   ├── true_news.csv       # (local only – not committed)
│   ├── fake_model.pkl      # (generated – not committed)
│   └── vectorizer.pkl      # (generated – not committed)
├── static/
│   └── style.css           # Dark-theme stylesheet
├── templates/
│   └── index.html          # Jinja2 template
└── tests/
    └── test_app.py         # Pytest test suite
```

---

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `FLASK_DEBUG` | `0` | Set to `1` to enable Flask debug/reloader |
| `PORT` | `5000` | Port the server listens on |

Example `.env` (never commit this file):

```
FLASK_DEBUG=1
PORT=5000
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes and add/update tests in `tests/`.
3. Ensure `pytest tests/ -v` passes before opening a pull request.
4. Open a PR against `main` with a clear description of your changes.
