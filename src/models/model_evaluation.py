import pickle
import pandas as pd
import json
import logging
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Configure logging
logging.basicConfig(
    filename="logs/model_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(path: str) -> object:
    """Load a trained model from disk."""
    try:
        with open(path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logging.error("Error loading model from %s: %s", path, e)
        raise

def load_test_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature-engineered test data and extract features and labels."""
    try:
        df = pd.read_csv(path)
        X_test = df.drop(columns=['sentiment']).values
        y_test = df['sentiment'].values
        logging.info("Test data loaded from %s", path)
        return X_test, y_test
    except Exception as e:
        logging.error("Error loading test data from %s: %s", path, e)
        raise

def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model's performance using various metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
        logging.info("Model evaluation completed")
        return metrics
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logging.info("Evaluation metrics saved to %s", path)
    except Exception as e:
        logging.error("Error saving metrics to %s: %s", path, e)
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        X_test, y_test = load_test_data("data/interim/test_tfidf.csv")
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/evaluation_metrics.json")
        logging.info("Model evaluation pipeline completed successfully")
    except Exception as e:
        logging.error("Model evaluation pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()  