from datetime import datetime
from pathlib import Path

import feast
import pandas as pd
import typer
from joblib import dump
from sklearn.linear_model import LinearRegression

# Load driver order data
# orders = pd.read_csv("driver_orders.csv", sep="\t")
# orders["event_timestamp"] = pd.to_datetime(orders["event_timestamp"])

# Connect to your local feature store
fs = feast.FeatureStore(repo_path="features/")


def get_dataset() -> pd.DataFrame:
    entity_df = pd.DataFrame.from_dict(
        {
            "driver_id": [1001, 1002, 1003, 1004, 1002, 1003, 1001],
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 15, 1, 12),
                datetime(2021, 4, 12, 10, 59, 42),
                datetime.now(),
            ],
            "trip_completed": [1, 0, 1, 0, 1, 0, 1],
        }
    )
    return entity_df


def add_features(training_df: pd.DataFrame) -> pd.DataFrame:
    training_df_with_features = fs.get_historical_features(
        entity_df=training_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()
    return training_df_with_features


def train_model(model_resutl_path: Path = Path("driver_model.bin")):
    training_df = get_dataset()
    training_df_with_features = add_features(training_df=training_df)
    print(f"training_df = {training_df_with_features.head()}")
    # Train model
    target = "trip_completed"

    reg = LinearRegression()
    train_X = training_df_with_features[training_df_with_features.columns.drop(target).drop("event_timestamp")]
    train_Y = training_df_with_features.loc[:, target]
    reg.fit(train_X[sorted(train_X)], train_Y)

    # Save model
    dump(reg, model_resutl_path)


if __name__ == "__main__":
    typer.run(train_model)
