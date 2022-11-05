from typing import List, Optional

import feast
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    model_path: str = "driver_model.bin"
    fs_repo_path: str = "features/"


class Payload(BaseModel):
    driver_ids: List[int]


class Prediction(BaseModel):
    best_driver_id: Optional[int] = None


class DriverRankingModel:
    def __init__(self, model_path: str, fs_repo_path: str):
        self.model = load(model_path)
        self.fs = feast.FeatureStore(repo_path=fs_repo_path)

    def predict(self, driver_ids: List[int]):
        # driver_ids = [1001, 1002, 1003, 1004, 2]

        df = self.fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
        ).to_df()
        print(df)

        df = df.dropna()
        if df.shape[0] == 0:
            return Prediction(best_driver_id=None)

        df["prediction"] = self.model.predict(df.dropna())
        best_driver_id = df["driver_id"].iloc[df["prediction"].argmax()]

        return Prediction(best_driver_id=best_driver_id)


settings = Settings()
model = DriverRankingModel(model_path=settings.model_path, fs_repo_path=settings.fs_repo_path)
app = FastAPI()


@app.get("/health_check")
def health_check() -> str:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = model.predict(driver_ids=payload.driver_ids)
    return prediction
