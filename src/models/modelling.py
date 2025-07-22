import numpy as np
import pandas as pd
import yaml
import pickle
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import os

# Configure logging
logging.basicConfig(
    filename="logs/modelling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params
    except Exception as e:
        logging.error("Error loading parameters: %s", e)
        raise

def load_training_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature-engineered training data and extract features and labels."""
    try:
        df = pd.read_csv(path)
        X = df.drop(columns=['sentiment']).values
        y = df['sentiment'].values
        logging.info("Training data loaded from %s", path)
        return X, y
    except Exception as e:
        logging.error("Error loading training data from %s: %s", path, e)
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Initialize and train the Random Forest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def save_model(model: RandomForestClassifier, path: str) -> None:
    """Save the trained model to disk using pickle."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as model_file:
            pickle.dump(model, model_file)
        logging.info("Model saved to %s", path)
    except Exception as e:
        logging.error("Error saving model to %s: %s", path, e)
        raise

def main() -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params("params.yaml")
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        X_train, y_train = load_training_data("data/interim/train_tfidf.csv")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully")
    except Exception as e:
        logging.error("Model training pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()