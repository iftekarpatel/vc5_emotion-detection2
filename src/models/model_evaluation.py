from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json

# Load the trained Random Forest model from disk
with open("models/random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load feature-engineered test data
test_data = pd.read_csv("data/interim/test_bow.csv")

# Extract features and labels from test data
X_test = test_data.drop(columns=['sentiment']).values  # Features for testing
y_test = test_data['sentiment'].values                 # Labels for testing

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Save evaluation metrics to a JSON file for later analysis
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall
}

with open("reports/evaluation_metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)