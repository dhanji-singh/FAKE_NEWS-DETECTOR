"""Tests for the Cyber Kavach Fake News Detector Flask application.

These tests run without requiring trained model artifacts by using a
lightweight mock model and vectorizer injected into the app module.
"""
import pickle

import pytest

import app as app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeVectorizer:
    """Minimal sklearn-compatible TF-IDF stub."""

    def transform(self, texts):
        # Return a simple list so the model stub receives something iterable.
        return texts


class _FakeModel:
    """Always predicts 'Real News' (label 1) with 80% confidence."""

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture(autouse=True)
def inject_mock_artifacts(monkeypatch):
    """Replace the global model/vectorizer with lightweight stubs."""
    monkeypatch.setattr(app_module, "model", _FakeModel())
    monkeypatch.setattr(app_module, "vectorizer", _FakeVectorizer())
    monkeypatch.setattr(app_module, "load_error", None)


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

class TestHomePage:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_title(self, client):
        response = client.get("/")
        assert b"Cyber Kavach" in response.data

    def test_contains_form(self, client):
        response = client.get("/")
        assert b"<form" in response.data
        assert b'action="/predict"' in response.data


# ---------------------------------------------------------------------------
# Predict endpoint – happy paths
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_returns_200(self, client):
        response = client.post("/predict", data={"news": "Some news headline"})
        assert response.status_code == 200

    def test_shows_prediction_label(self, client):
        response = client.post("/predict", data={"news": "Some news headline"})
        assert b"Real News" in response.data or b"Fake News" in response.data

    def test_shows_confidence(self, client):
        response = client.post("/predict", data={"news": "Some news headline"})
        assert b"Confidence:" in response.data

    def test_echoes_submitted_text(self, client):
        response = client.post("/predict", data={"news": "Echo this text"})
        assert b"Echo this text" in response.data


# ---------------------------------------------------------------------------
# Predict endpoint – validation / error paths
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_empty_input_returns_error(self, client):
        response = client.post("/predict", data={"news": ""})
        assert response.status_code == 200
        assert b"Please enter some news text." in response.data

    def test_whitespace_only_returns_error(self, client):
        response = client.post("/predict", data={"news": "   "})
        assert b"Please enter some news text." in response.data

    def test_oversized_input_returns_error(self, client):
        long_text = "a" * (app_module.MAX_INPUT_LENGTH + 1)
        response = client.post("/predict", data={"news": long_text})
        assert b"too long" in response.data.lower()

    def test_missing_news_field_returns_error(self, client):
        response = client.post("/predict", data={})
        assert b"Please enter some news text." in response.data


# ---------------------------------------------------------------------------
# Predict endpoint – model not loaded
# ---------------------------------------------------------------------------

class TestPredictModelNotLoaded:
    def test_shows_load_error_when_model_missing(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "load_error",
                            "Model files not found. Run: python model/train_model.py")
        response = client.post("/predict", data={"news": "Some text"})
        assert b"Model files not found" in response.data

    def test_home_shows_load_error(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "load_error", "Model files not found.")
        response = client.get("/")
        assert b"Model files not found." in response.data


# ---------------------------------------------------------------------------
# load_artifacts helper
# ---------------------------------------------------------------------------

class TestLoadArtifacts:
    def test_returns_error_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.pkl")
        monkeypatch.setattr(app_module, "VECTORIZER_PATH", tmp_path / "missing2.pkl")
        m, v, err = app_module.load_artifacts()
        assert m is None
        assert v is None
        assert "not found" in err.lower()

    def test_returns_error_on_corrupt_pickle(self, tmp_path, monkeypatch):
        bad_model = tmp_path / "bad_model.pkl"
        bad_vec = tmp_path / "bad_vec.pkl"
        bad_model.write_bytes(b"not a valid pickle")
        bad_vec.write_bytes(b"not a valid pickle")
        monkeypatch.setattr(app_module, "MODEL_PATH", bad_model)
        monkeypatch.setattr(app_module, "VECTORIZER_PATH", bad_vec)
        m, v, err = app_module.load_artifacts()
        assert m is None
        assert "failed to load" in err.lower()

    def test_loads_valid_artifacts(self, tmp_path, monkeypatch):
        model_path = tmp_path / "fake_model.pkl"
        vec_path = tmp_path / "vectorizer.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(_FakeModel(), f)
        with open(vec_path, "wb") as f:
            pickle.dump(_FakeVectorizer(), f)
        monkeypatch.setattr(app_module, "MODEL_PATH", model_path)
        monkeypatch.setattr(app_module, "VECTORIZER_PATH", vec_path)
        m, v, err = app_module.load_artifacts()
        assert m is not None
        assert v is not None
        assert err is None
