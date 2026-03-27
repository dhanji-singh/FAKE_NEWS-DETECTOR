import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve paths relative to this script so execution works from any CWD.
BASE_DIR = Path(__file__).resolve().parent
FAKE_NEWS_PATH = BASE_DIR / "fake_news.csv"
TRUE_NEWS_PATH = BASE_DIR / "true_news.csv"
MODEL_PATH = BASE_DIR / "fake_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def read_news_csv(path: Path) -> pd.DataFrame:
    """Read CSV with a fallback encoding for datasets that contain mixed characters."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main() -> None:
    logger.info("Loading datasets …")

    # Build labeled dataset from separate fake/true files.
    fake_df = read_news_csv(FAKE_NEWS_PATH)[["text"]].copy()
    fake_df["label"] = 0

    true_df = read_news_csv(TRUE_NEWS_PATH)[["text"]].copy()
    true_df["label"] = 1

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str)

    logger.info("Dataset size: %d samples (%d fake, %d real)",
                len(data), (data["label"] == 0).sum(), (data["label"] == 1).sum())

    X = data["text"]
    y = data["label"]

    # Vectorization
    logger.info("Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    logger.info("Training Logistic Regression model …")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Evaluation
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info("Train accuracy: %.4f | Test accuracy: %.4f", train_acc, test_acc)
    logger.info("Classification report:\n%s",
                classification_report(y_test, clf.predict(X_test),
                                      target_names=["Fake", "Real"]))

    # Save model + vectorizer
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(clf, model_file)
    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    logger.info("Saved model → %s", MODEL_PATH)
    logger.info("Saved vectorizer → %s", VECTORIZER_PATH)


if __name__ == "__main__":
    main()
