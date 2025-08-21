import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Flag missing Cabin values and create binary indicator
    df['Cabin_Missing'] = df['Cabin'].isna().astype(int)
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    return df

def engineer_features(df):
    # Create IsChild feature (age < 16)
    df['IsChild'] = (df['Age'] < 16).astype(int)
    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def encode_categorical(df):
    le = LabelEncoder()
    # Encode categorical variables
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title'] = le.fit_transform(df['Title'])
    # Drop columns not needed for modeling
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

def main():
    # File paths
    input_file = 'data/train.csv'
    output_file = 'data/titanic_clean.csv'
    
    # Process data
    df = load_data(input_file)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_categorical(df)
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    main()