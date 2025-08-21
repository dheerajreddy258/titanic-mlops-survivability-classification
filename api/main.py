from fastapi import FastAPI
from .schemas import TitanicFeatures
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load the saved model
model_path = [f for f in os.listdir('models') if f.startswith('best_model_')][0]
model = joblib.load(f'models/{model_path}')

# Initialize LabelEncoder for preprocessing
le_sex = LabelEncoder().fit(['male', 'female'])
le_embarked = LabelEncoder().fit(['C', 'Q', 'S'])
le_title = LabelEncoder().fit(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])

def preprocess_features(features: TitanicFeatures):
    # Convert to DataFrame
    df = pd.DataFrame([features.dict()])
    
    # Encode categorical variables (same as prep_data.py)
    df['Sex'] = le_sex.transform(df['Sex'])
    df['Embarked'] = le_embarked.transform(df['Embarked'])
    df['Title'] = le_title.transform(df['Title'])
    
    # Ensure correct feature order
    feature_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                     'Cabin_Missing', 'IsChild', 'FamilySize', 'IsAlone', 'Title']
    df = df[feature_order]
    return df

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict(features: TitanicFeatures):
    # Preprocess input
    X = preprocess_features(features)
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Probability of surviving (class 1)
    
    return {
        "survived": int(prediction),
        "probability": float(probability)
    }