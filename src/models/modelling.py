import numpy as np
import pandas as pd 

import pickle

from sklearn.ensemble import RandomForestClassifier

# Load feature-engineered training data
train_data = pd.read_csv("data/interim/train_bow.csv")
x_train = train_data.drop(columns=['sentiment']).values  # Features for training
y_train = train_data['sentiment'].values                 # Labels for training

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Save the trained model to disk using pickle
with open("models/random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)