# 🚢 Titanic Survivability Classification (Capstone Project)

This capstone project demonstrates a **complete MLOps pipeline** for the
famous Titanic dataset.\
The main goal is to build and deploy a **binary classification model**
predicting passenger survival (`Survived` vs `Not Survived`).

------------------------------------------------------------------------

## 🎯 Objectives

-   End-to-end MLOps pipeline on local machine\
-   Data versioning and preprocessing\
-   Experiment tracking with **MLflow**\
-   API deployment using **FastAPI**\
-   Containerization with **Docker**\
-   CI/CD pipeline with **GitHub Actions**\
-   Logging & Monitoring (SQLite, Python logging)  
------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── data/
    │   └── train.csv                
    ├── src/
    │   ├── prep_data.py             # data cleaning & feature engineering
    │   ├── train.py                 # model training + MLflow logging
    │   └── evaluate.py              # model evaluation
    ├── api/
    │   ├── main.py                  # FastAPI service
    │   └── schemas.py               # Pydantic schemas for validation
    ├── models/                      # trained models & artifacts
    ├── tests/                       # unit tests
    ├── requirements.txt             # dependencies
    ├── Dockerfile                   # container definition
    ├── .github/
    │   └── workflows/ci.yml         # GitHub Actions CI/CD
    └── README.md

------------------------------------------------------------------------

## ⚙️ Setup Instructions

1.  **Clone the repository**

    ``` bash
    git clone <your-repo-url>
    cd titanic-capstone
    ```

2.  **Create a virtual environment**

    ``` bash
    python -m venv .venv
    source .venv/bin/activate     # Linux/Mac
    # .venv\Scripts\activate    # Windows
    ```

3.  **Install dependencies**

    ``` bash
    pip install -r requirements.txt
    ```

4.  **Download dataset** Get `train.csv` from [Kaggle Titanic
    Dataset](https://www.kaggle.com/c/titanic/data) and place it in the
    `data/` folder.

------------------------------------------------------------------------

## 🧹 Part 1: Data Preparation

Run preprocessing:

``` bash
python src/prep_data.py --input data/train.csv --output data/titanic_clean.csv
```

-   Handles missing values (Age, Cabin, Embarked, Fare)\
-   Engineers features (`IsChild`, `FamilySize`, `IsAlone`, `HasCabin`,
    `Title`)\
-   Encodes categorical variables (`Sex`, `Embarked`, `Title`,
    `TicketGroup`)\
-   Outputs cleaned dataset to `data/titanic_clean.csv`

------------------------------------------------------------------------

## 🤖 Part 2: Model Development & Experiment Tracking

Models used: - Logistic Regression\
- Random Forest Classifier\
- Gradient Boosting Classifier

With **MLflow**, we log: - Model parameters\
- Training metrics (Accuracy, Precision, Recall, F1)\
- Feature importance\
- Best model registration in the model registry

Run training:

``` bash
python src/train.py
```

Evaluate:

``` bash
python src/evaluate.py
```

------------------------------------------------------------------------

## 🌐 Part 3: API & Docker Packaging

FastAPI endpoints: - `GET /health` → returns status `"OK"`\
- `POST /predict` → accepts passenger features JSON and returns survival
prediction & probability

Example **Pydantic schema** (`schemas.py`):

``` python
class TitanicFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Cabin_Missing: Optional[int] = 0
    IsChild: Optional[int] = 0
    FamilySize: Optional[int] = 1
    IsAlone: Optional[int] = 1
    Title: Optional[str] = "Mr"
```

Run locally:

``` bash
uvicorn api.main:app --reload
```

Docker build & run:

``` bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

DockerHub repo: 👉
[breadkrum/titanic](https://hub.docker.com/r/breadkrum/titanic/tags)

------------------------------------------------------------------------

## 🔄 Part 4: CI/CD with GitHub Actions

CI/CD workflow (`.github/workflows/ci.yml`): - Lint & test with
**flake8** + **pytest**\
- Build Docker image `titanic-api`\
- Push image to DockerHub
([breadkrum/titanic](https://hub.docker.com/r/breadkrum/titanic/tags))\
- Deploy (optional via self-hosted runner)

------------------------------------------------------------------------

## 📊 Part 5: Logging & Monitoring

-   Each API request is logged with:
    -   Features\
    -   Timestamp\
    -   Prediction & probability\
    -   Latency
-   Logs are persisted to:
    -   Daily rotating log file\
    -   SQLite database (predictions table)




👨‍💻 Developed as a Capstone Project for hands-on learning in **AI/ML &
MLOps**.
