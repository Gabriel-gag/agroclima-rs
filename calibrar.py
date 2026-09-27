from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLD_PATH = BASE_DIR / "data" / "gold" / "features_inundacao.parquet"
MODEL_PATH = BASE_DIR / "src" / "model_calibrado.joblib"


# ============================================================
# FEATURES
# ============================================================

MODEL_FEATURES = [
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
    "altitude",
    "latitude",
    "longitude",
]


# ============================================================
# 1. CARREGAR GOLD
# ============================================================

print("Carregando Gold...")

df = pd.read_parquet(GOLD_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df[df["inundacao_t1"].notna()].copy()

df["inundacao_t1"] = df["inundacao_t1"].astype(int)

df = df.sort_values(
    ["date", "station_id"]
).reset_index(drop=True)

print(f"Registros válidos: {len(df):,}")
print(f"Positivos: {df['inundacao_t1'].sum():,}")


# ============================================================
# 2. SPLIT TEMPORAL
#
# TREINO:       2006–2019
# CALIBRAÇÃO:   2020–2023
# HOLDOUT:      2024–2025
# ============================================================

df_treino = df[
    df["date"].dt.year <= 2019
].copy()

df_calibracao = df[
    df["date"].dt.year.between(2020, 2023)
].copy()

df_holdout = df[
    df["date"].dt.year.between(2024, 2025)
].copy()


print("\n============================================")
print("DIVISÃO TEMPORAL")
print("============================================")

print(
    f"Treino:      {len(df_treino):,} registros | "
    f"{df_treino['inundacao_t1'].sum()} positivos"
)

print(
    f"Calibração:  {len(df_calibracao):,} registros | "
    f"{df_calibracao['inundacao_t1'].sum()} positivos"
)

print(
    f"Holdout:     {len(df_holdout):,} registros | "
    f"{df_holdout['inundacao_t1'].sum()} positivos"
)


# ============================================================
# 3. SEPARAR X E Y
# ============================================================

X_treino = df_treino[MODEL_FEATURES]
y_treino = df_treino["inundacao_t1"]

X_cal = df_calibracao[MODEL_FEATURES]
y_cal = df_calibracao["inundacao_t1"]

X_holdout = df_holdout[MODEL_FEATURES]
y_holdout = df_holdout["inundacao_t1"]


# ============================================================
# 4. MODELO BASE
# ============================================================

print("\nTreinando regressão logística...")

modelo_base = Pipeline(
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

modelo_base.fit(X_treino, y_treino)


# ============================================================
# 5. CALIBRAÇÃO
# ============================================================

print("Calibrando probabilidades...")

modelo_calibrado = CalibratedClassifierCV(
    estimator=modelo_base,
    method="sigmoid",
    cv="prefit"
)

modelo_calibrado.fit(
    X_cal,
    y_cal
)


# ============================================================
# 6. AVALIAR HOLDOUT
# ============================================================

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)

probabilidades = modelo_calibrado.predict_proba(
    X_holdout
)[:, 1]


pr_auc = average_precision_score(
    y_holdout,
    probabilidades
)

roc_auc = roc_auc_score(
    y_holdout,
    probabilidades
)

brier = brier_score_loss(
    y_holdout,
    probabilidades
)


print("\n============================================")
print("RESULTADOS — HOLDOUT 2024–2025")
print("============================================")

print(f"PR-AUC:  {pr_auc:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")
print(f"Brier:   {brier:.6f}")


# ============================================================
# 7. MOSTRAR EVENTOS REAIS
# ============================================================

resultado = df_holdout[
    [
        "date",
        "station_id",
        "municipio",
        "inundacao_t1"
    ]
].copy()

resultado["probabilidade"] = probabilidades

eventos = resultado[
    resultado["inundacao_t1"] == 1
].sort_values(
    "probabilidade",
    ascending=False
)

print("\n============================================")
print("EVENTOS REAIS NO HOLDOUT")
print("============================================")

print(
    eventos.to_string(index=False)
)


# ============================================================
# 8. SALVAR MODELO
# ============================================================

artifact = {
    "features": MODEL_FEATURES,
    "model": modelo_calibrado,
    "target": "inundacao_t1",
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print("\n============================================")
print("MODELO SALVO")
print("============================================")

print(MODEL_PATH)