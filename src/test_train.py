import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from train import load_data, train_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@pytest.fixture
def cleaned_data():
    return load_data('data/titanic_clean.csv')  # Use 'data/train_clean.csv' if different

def test_load_cleaned_data(cleaned_data):
    assert cleaned_data.shape[0] > 0, "Cleaned dataset should have rows"
    assert 'Survived' in cleaned_data.columns, "Survived column should exist"

def test_train_model(cleaned_data):
    X = cleaned_data.drop('Survived', axis=1)
    y = cleaned_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    trained_model, metrics = train_model(model, X_train, y_train, X_test, y_test)
    assert metrics['accuracy'] > 0.5, "Model accuracy should be above 0.5"
    assert 'precision' in metrics, "Metrics should include precision"
