import pickle
import os
from sklearn.linear_model import LogisticRegression
import numpy as np

MODEL_PATH = "models/signal_model.pkl"

def train_model(X, y):
    """Train a simple logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model

def load_model(path=MODEL_PATH):
    """Safely load a model from pickle. Returns None if missing or broken."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load model ({e}). Will retrain.")
        return None

def predict_signal(model, X):
    """Return trading signal and confidence."""
    if model is None:
        return "HOLD", 0.5
    # Ensure no NaNs
    X_clean = X.fillna(0)
    proba = model.predict_proba(X_clean)[0][1]
    signal = "BUY" if proba >= 0.5 else "SELL"
    return signal, proba
