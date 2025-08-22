import pandas as pd
import os
import pytest
from prep_data import load_data, handle_missing_values, engineer_features, encode_categorical

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'PassengerId': [1],
        'Survived': [0],
        'Pclass': [3],
        'Name': ['Braund, Mr. Owen Harris'],
        'Sex': ['male'],
        'Age': [None],
        'SibSp': [1],
        'Parch': [0],
        'Ticket': ['A/5 21171'],
        'Fare': [7.25],
        'Cabin': [None],
        'Embarked': ['S']
    })

def test_load_data():
    if os.path.exists('data/train.csv'):
        df = load_data('data/train.csv')
        assert df.shape[0] == 891, "Dataset should have 891 rows"
    else:
        pytest.skip("train.csv not found for testing")

def test_handle_missing_values(sample_data):
    df = handle_missing_values(sample_data)
    assert df['Age'].isna().sum() == 0, "Age should have no missing values"
    assert df['Cabin_Missing'][0] == 1, "Cabin_Missing should be 1"

def test_engineer_features(sample_data):
    df = handle_missing_values(sample_data)  # Pre-step
    df = engineer_features(df)
    assert 'IsChild' in df.columns, "IsChild feature should be created"
    assert df['FamilySize'][0] == 2, "FamilySize should be 2"

def test_encode_categorical(sample_data):
    df = handle_missing_values(sample_data)  # Pre-step
    df = engineer_features(df)  # Pre-step
    df = encode_categorical(df)
    assert 'Sex' in df.columns and isinstance(df['Sex'][0], int), "Sex should be encoded as int"
    assert 'PassengerId' not in df.columns, "PassengerId should be dropped"