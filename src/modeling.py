from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

WEATHER_FEATURES = [
    "precip_mm",
    "precip_sum_3d",
    "precip_sum_7d",
    "precip_sum_24h",
    "precip_sum_48h",
    "precip_sum_72h",
    "rain_days_7d",
    "dry_days_7d",
    "temperature_min",
    "temperature_max",
    "temperature_avg",
    "temperature_avg_3d",
    "humidity_avg",
    "humidity_avg_3d",
    "pressure_avg",
    "pressure_change_24h",
    "wind_speed_avg",
    "solar_radiation",
    "season_sin",
    "season_cos",
]

WEATHER_ALTITUDE_FEATURES = WEATHER_FEATURES + [
    "altitude"
]

WEATHER_COORDINATES_FEATURES = WEATHER_FEATURES + [
    "latitude",
    "longitude"
]

MODEL_FEATURES = WEATHER_FEATURES + [
    "altitude",
    "latitude",
    "longitude"
]

def preparar_dataset_modelo(gold_path, features=None):
    gold_path = Path(gold_path)

    df = pd.read_parquet(gold_path)

    if features is None:
        features = MODEL_FEATURES

    required = [
        "date",
        "station_id",
        "inundacao_t1",
        *features,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colunas ausentes no Gold: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df[
        df["inundacao_t1"].notna()
    ].copy()

    df["inundacao_t1"] = (
        df["inundacao_t1"]
        .astype(int)
    )

    df = df.sort_values(
        ["date", "station_id"]
    ).reset_index(drop=True)

    return df


def dividir_temporalmente(df):

    treino = df[
        df["date"].dt.year <= 2023
    ].copy()

    holdout = df[
        df["date"].dt.year.between(2024, 2025)
    ].copy()

    return treino, holdout


def criar_modelo_logistico():

    modelo = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42
                )
            ),
        ]
    )

    return modelo

def criar_modelo_random_forest():

    modelo = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    return modelo