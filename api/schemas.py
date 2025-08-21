from pydantic import BaseModel
from typing import Optional

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