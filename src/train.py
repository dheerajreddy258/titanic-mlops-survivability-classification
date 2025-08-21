import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    return model, metrics

def main():
    # Load cleaned data
    df = load_data('data/titanic_clean.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    best_model = None
    best_accuracy = 0
    best_model_name = ''
    results = {}
    
    for name, model in models.items():
        trained_model, metrics = train_model(model, X_train, y_train, X_test, y_test)
        results[name] = metrics
        print(f"{name} Metrics: {metrics}")
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = trained_model
            best_model_name = name
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model_{best_model_name}.pkl')
    print(f"Best model ({best_model_name}) saved to models/best_model_{best_model_name}.pkl with accuracy: {best_accuracy}")
    
    # Print feature importance for tree-based models
    if best_model_name in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
        print(f"\nFeature Importance for {best_model_name}:")
        print(feature_importance.sort_values(ascending=False))

if __name__ == "__main__":
    main()