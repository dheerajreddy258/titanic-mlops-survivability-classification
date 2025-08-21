import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def evaluate_model(y_true, y_pred, model_name):
    print(f"Evaluation for {model_name}:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"  F1: {f1_score(y_true, y_pred):.4f}")

def main():
    # Load data
    data = load_data('data/titanic_clean.csv')
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Load best model (assumes one model exists in models/)
    model_path = [f for f in os.listdir('models') if f.endswith('.pkl')][0]
    model = joblib.load(f'models/{model_path}')
    model_name = model_path.replace('.pkl', '')
    
    # Ensure feature names match those used during training
    expected_features = model.feature_names_in_
    X = X[expected_features]
    
    # Predict and evaluate
    y_pred = model.predict(X)
    evaluate_model(y, y_pred, model_name)

if __name__ == "__main__":
    main()