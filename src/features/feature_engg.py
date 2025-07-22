import numpy as np
import pandas as pd
import logging
import yaml
import os
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    filename="logs/feature_engg.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params
    except Exception as e:
        logging.error("Error loading parameters: %s", e)
        raise

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file and drop rows with missing 'content'."""
    try:
        df = pd.read_csv(path).dropna(subset=['content'])
        logging.info("Data loaded from %s", path)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", path, e)
        raise

def extract_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from DataFrame."""
    try:
        X = df['content'].values
        y = df['sentiment'].values
        logging.info("Features and labels extracted")
        return X, y
    except Exception as e:
        logging.error("Error extracting features and labels: %s", e)
        raise

def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Vectorize text data using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logging.info("Text data vectorized using TF-IDF")
        return X_train_tfidf, X_test_tfidf, vectorizer
    except Exception as e:
        logging.error("Error vectorizing data: %s", e)
        raise

def save_feature_data(X_bow, y, path: str) -> None:
    """Save feature-engineered data to CSV."""
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['sentiment'] = y
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("Feature data saved to %s", path)
    except Exception as e:
        logging.error("Error saving feature data to %s: %s", path, e)
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engg']['max_features']

        train_data = load_data("data/processed/train.csv")
        test_data = load_data("data/processed/test.csv")

        X_train, y_train = extract_features_and_labels(train_data)
        X_test, y_test = extract_features_and_labels(test_data)

        X_train_tfidf, X_test_tfidf, _ = vectorize_data(X_train, X_test, max_features)

        save_feature_data(X_train_tfidf, y_train, "data/interim/train_tfidf.csv")
        save_feature_data(X_test_tfidf, y_test, "data/interim/test_tfidf.csv")

        logging.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logging.error("Feature engineering pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()