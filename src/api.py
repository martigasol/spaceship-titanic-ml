from fastapi import FastAPI
import joblib
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.features.engineering import engineer_features


class Passenger(BaseModel):
    PassengerId: Optional[str] = None
    HomePlanet: Optional[str] = None
    CryoSleep: Optional[bool] = None
    Destination: Optional[str] = None
    Age: Optional[float] = None
    VIP: Optional[bool] = None
    RoomService: Optional[float] = None
    FoodCourt: Optional[float] = None
    ShoppingMall: Optional[float] = None
    Spa: Optional[float] = None
    VRDeck: Optional[float] = None
    Cabin: Optional[str] = None

app = FastAPI(
    title="Spaceship Titanic ML API",
    description="API for predicting whether a passenger was transported.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "spaceship_titanic_catboost.joblib"

model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(passenger: Passenger):
    passenger_df = pd.DataFrame([passenger.model_dump()])

    passenger_df = engineer_features(passenger_df)

    passenger_df = passenger_df.drop(
        columns=["PassengerId", "Name", "Cabin"],
        errors="ignore",
    )

    prediction = model.predict(passenger_df)[0]

    return {
        "Transported": bool(prediction)
    }

#http://127.0.0.1:8000/docs
#uvicorn src.api:app --reload
