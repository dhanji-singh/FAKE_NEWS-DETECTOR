import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

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


# Build labeled dataset from separate fake/true files.
fake_df = read_news_csv(FAKE_NEWS_PATH)[["text"]].copy()
fake_df["label"] = 0

true_df = read_news_csv(TRUE_NEWS_PATH)[["text"]].copy()
true_df["label"] = 1

data = pd.concat([fake_df, true_df], ignore_index=True)
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)

X = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Save model + vectorizer
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)