from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLD_PATH = BASE_DIR / "data" / "gold" / "features_inundacao.parquet"
MODEL_PATH = BASE_DIR / "src" / "model_calibrado.joblib"
RESULTS_PATH = BASE_DIR / "data" / "gold" / "validacao_progressiva.csv"


# ============================================================
# CONFIGURAÇÃO DA VALIDAÇÃO
# ============================================================

GAP_DAYS = 30

# Quantos anos serão utilizados exclusivamente
# para calibração antes do gap.
CALIBRATION_YEARS = 4

# Primeiro ano que será utilizado como teste.
FIRST_TEST_YEAR = 2020

# Último ano disponível para validação.
LAST_TEST_YEAR = 2025


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
# FUNÇÃO PARA CRIAR O MODELO
# ============================================================

def criar_modelo():

    return Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# 1. CARREGAR GOLD
# ============================================================

print("Carregando Gold...")

df = pd.read_parquet(GOLD_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
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


print("\n============================================")
print("BASE")
print("============================================")

print(
    f"Período: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print(
    f"Registros válidos: {len(df):,}"
)

print(
    f"Positivos: {df['inundacao_t1'].sum():,}"
)


# ============================================================
# 2. FUNÇÃO DE VALIDAÇÃO PROGRESSIVA
# ============================================================

resultados = []


for test_year in range(
    FIRST_TEST_YEAR,
    LAST_TEST_YEAR + 1
):

    # --------------------------------------------------------
    # PERÍODO DE TESTE
    # --------------------------------------------------------

    inicio_teste = pd.Timestamp(
        f"{test_year}-01-01"
    )

    fim_teste = pd.Timestamp(
        f"{test_year}-12-31"
    )

    # --------------------------------------------------------
    # GAP
    #
    # O treinamento/calibração não podem utilizar
    # informações dos 30 dias anteriores ao teste.
    # --------------------------------------------------------

    fim_desenvolvimento = (
        inicio_teste
        - pd.Timedelta(days=GAP_DAYS)
    )

    # --------------------------------------------------------
    # PERÍODO DE CALIBRAÇÃO
    #
    # Utilizamos os 4 anos imediatamente anteriores
    # ao gap.
    # --------------------------------------------------------

    inicio_calibracao = (
        fim_desenvolvimento
        - pd.DateOffset(
            years=CALIBRATION_YEARS
        )
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------------
    # TREINO
    #
    # Tudo anterior ao período de calibração.
    # --------------------------------------------------------

    df_treino = df[
        df["date"] < inicio_calibracao
    ].copy()

    # --------------------------------------------------------
    # CALIBRAÇÃO
    # --------------------------------------------------------

    df_calibracao = df[
        (
            df["date"] >= inicio_calibracao
        )
        &
        (
            df["date"] <= fim_desenvolvimento
        )
    ].copy()

    # --------------------------------------------------------
    # TESTE
    #
    # Somente o ano correspondente.
    # --------------------------------------------------------

    df_teste = df[
        (
            df["date"] >= inicio_teste
        )
        &
        (
            df["date"] <= fim_teste
        )
    ].copy()

    # --------------------------------------------------------
    # VERIFICAÇÃO
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print(f"FOLD {test_year}")
    print("============================================")

    print(
        f"Treino:      "
        f"{df_treino['date'].min().date()} → "
        f"{df_treino['date'].max().date()}"
    )

    print(
        f"Calibração:  "
        f"{df_calibracao['date'].min().date()} → "
        f"{df_calibracao['date'].max().date()}"
    )

    print(
        f"Gap:         "
        f"{fim_desenvolvimento.date()} → "
        f"{(inicio_teste - pd.Timedelta(days=1)).date()}"
    )

    print(
        f"Teste:       "
        f"{df_teste['date'].min().date()} → "
        f"{df_teste['date'].max().date()}"
    )

    print(
        f"Positivos treino: "
        f"{df_treino['inundacao_t1'].sum()}"
    )

    print(
        f"Positivos calibração: "
        f"{df_calibracao['inundacao_t1'].sum()}"
    )

    print(
        f"Positivos teste: "
        f"{df_teste['inundacao_t1'].sum()}"
    )

    # --------------------------------------------------------
    # VERIFICAR SE EXISTEM POSITIVOS
    # --------------------------------------------------------

    if df_treino["inundacao_t1"].nunique() < 2:

        print(
            "Fold ignorado: treino possui apenas "
            "uma classe."
        )

        continue

    if df_calibracao["inundacao_t1"].nunique() < 2:

        print(
            "Fold ignorado: calibração possui apenas "
            "uma classe."
        )

        continue

    if df_teste["inundacao_t1"].nunique() < 2:

        print(
            "Aviso: teste possui apenas uma classe. "
            "Algumas métricas podem não ser calculáveis."
        )

    # --------------------------------------------------------
    # 3. SEPARAR X E Y
    # --------------------------------------------------------

    X_treino = df_treino[
        MODEL_FEATURES
    ]

    y_treino = df_treino[
        "inundacao_t1"
    ]

    X_cal = df_calibracao[
        MODEL_FEATURES
    ]

    y_cal = df_calibracao[
        "inundacao_t1"
    ]

    X_teste = df_teste[
        MODEL_FEATURES
    ]

    y_teste = df_teste[
        "inundacao_t1"
    ]

    # --------------------------------------------------------
    # 4. TREINAR MODELO
    # --------------------------------------------------------

    print(
        "\nTreinando regressão logística..."
    )

    modelo_base = criar_modelo()

    modelo_base.fit(
        X_treino,
        y_treino
    )

    # --------------------------------------------------------
    # 5. CALIBRAR
    # --------------------------------------------------------
    print("Modelo:", type(modelo_base))
    print("X_cal:", X_cal.shape)
    print("Positivos:", y_cal.sum())
    print("Negativos:", (y_cal == 0).sum())

    print(
        "Calibrando probabilidades..."
    )

    modelo_calibrado = CalibratedClassifierCV(
        estimator=FrozenEstimator(modelo_base),
        method="sigmoid",
    )

    modelo_calibrado.fit(
        X_cal,
        y_cal,
    )
    print("Calibrador:", type(modelo_calibrado))

    print(modelo_calibrado)

    # --------------------------------------------------------
    # 6. GERAR PROBABILIDADES
    # --------------------------------------------------------

    probabilidades = (
        modelo_calibrado
        .predict_proba(X_teste)[:, 1]
    )

    # --------------------------------------------------------
    # 7. MÉTRICAS
    # --------------------------------------------------------

    try:

        pr_auc = average_precision_score(
            y_teste,
            probabilidades,
        )

    except ValueError:

        pr_auc = None

    try:

        roc_auc = roc_auc_score(
            y_teste,
            probabilidades,
        )

    except ValueError:

        roc_auc = None

    brier = brier_score_loss(
        y_teste,
        probabilidades,
    )

    # --------------------------------------------------------
    # 8. RESULTADO DO FOLD
    # --------------------------------------------------------

    resultado_fold = {

        "ano_teste": test_year,

        "inicio_treino":
            df_treino["date"].min(),

        "fim_treino":
            df_treino["date"].max(),

        "inicio_calibracao":
            df_calibracao["date"].min(),

        "fim_calibracao":
            df_calibracao["date"].max(),

        "inicio_teste":
            df_teste["date"].min(),

        "fim_teste":
            df_teste["date"].max(),

        "gap_dias":
            GAP_DAYS,

        "registros_treino":
            len(df_treino),

        "positivos_treino":
            int(
                y_treino.sum()
            ),

        "registros_calibracao":
            len(df_calibracao),

        "positivos_calibracao":
            int(
                y_cal.sum()
            ),

        "registros_teste":
            len(df_teste),

        "positivos_teste":
            int(
                y_teste.sum()
            ),

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "brier":
            brier,
    }

    resultados.append(
        resultado_fold
    )

    print("\nRESULTADO")

    print(
        f"PR-AUC:  {pr_auc}"
    )

    print(
        f"ROC-AUC: {roc_auc}"
    )

    print(
        f"Brier:   {brier}"
    )


# ============================================================
# 9. SALVAR RESULTADOS
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df_resultados.to_csv(
    RESULTS_PATH,
    index=False,
)


print("\n============================================")
print("VALIDAÇÃO PROGRESSIVA")
print("============================================")

print(
    df_resultados[
        [
            "ano_teste",
            "positivos_teste",
            "pr_auc",
            "roc_auc",
            "brier",
        ]
    ].to_string(index=False)
)

print(
    f"\nResultados salvos em:"
    f"\n{RESULTS_PATH}"
)


# ============================================================
# 10. TREINAR MODELO FINAL
# ============================================================

print("\n============================================")
print("MODELO FINAL")
print("============================================")

# ------------------------------------------------------------
# DATA DA PRIMEIRA PREVISÃO
# ------------------------------------------------------------

FORECAST_START = pd.Timestamp("2026-09-27")

# ------------------------------------------------------------
# GAP DE 30 DIAS
# ------------------------------------------------------------

data_fim_desenvolvimento = (
    FORECAST_START
    - pd.Timedelta(days=GAP_DAYS)
)

# ------------------------------------------------------------
# CALIBRAÇÃO FINAL
#
# Últimos 2 anos disponíveis ANTES do gap.
# ------------------------------------------------------------

inicio_cal_final = (
    data_fim_desenvolvimento
    - pd.DateOffset(years=CALIBRATION_YEARS)
    + pd.Timedelta(days=1)
)

# ------------------------------------------------------------
# TREINO FINAL
#
# Tudo anterior à calibração.
# ------------------------------------------------------------

df_treino_final = df[
    df["date"] < inicio_cal_final
].copy()

# ------------------------------------------------------------
# CALIBRAÇÃO FINAL
# ------------------------------------------------------------

df_cal_final = df[
    (
        df["date"] >= inicio_cal_final
    )
    &
    (
        df["date"] <= data_fim_desenvolvimento
    )
].copy()

print(
    f"Primeira previsão: "
    f"{FORECAST_START.date()}"
)

print(
    f"Fim do desenvolvimento: "
    f"{data_fim_desenvolvimento.date()}"
)

print(
    f"Treino final: "
    f"{df_treino_final['date'].min().date()} → "
    f"{df_treino_final['date'].max().date()}"
)

print(
    f"Calibração final: "
    f"{df_cal_final['date'].min().date()} → "
    f"{df_cal_final['date'].max().date()}"
)

print(
    f"Gap: "
    f"{(data_fim_desenvolvimento + pd.Timedelta(days=1)).date()} → "
    f"{(FORECAST_START - pd.Timedelta(days=1)).date()}"
)

print(
    f"Registros treino: "
    f"{len(df_treino_final):,}"
)

print(
    f"Positivos treino: "
    f"{df_treino_final['inundacao_t1'].sum()}"
)

print(
    f"Registros calibração: "
    f"{len(df_cal_final):,}"
)

print(
    f"Positivos calibração: "
    f"{df_cal_final['inundacao_t1'].sum()}"
)

# ------------------------------------------------------------
# VERIFICAR CLASSES
# ------------------------------------------------------------

if df_treino_final["inundacao_t1"].nunique() < 2:
    raise ValueError(
        "O treino final possui apenas uma classe."
    )

if df_cal_final["inundacao_t1"].nunique() < 2:
    raise ValueError(
        "A calibração final possui apenas uma classe."
    )

# ------------------------------------------------------------
# SEPARAR X E Y
# ------------------------------------------------------------

X_final = df_treino_final[
    MODEL_FEATURES
]

y_final = df_treino_final[
    "inundacao_t1"
]

X_cal_final = df_cal_final[
    MODEL_FEATURES
]

y_cal_final = df_cal_final[
    "inundacao_t1"
]

# ------------------------------------------------------------
# TREINAR MODELO BASE
# ------------------------------------------------------------

print(
    "\nTreinando modelo final..."
)

modelo_final = criar_modelo()

modelo_final.fit(
    X_final,
    y_final
)

# ============================================================
# 11. CALIBRAR MODELO FINAL
# ============================================================

print(
    "Calibrando probabilidades finais..."
)

modelo_calibrado_final = CalibratedClassifierCV(
    estimator=FrozenEstimator(modelo_final),
    method="sigmoid",
)

modelo_calibrado_final.fit(
    X_cal_final,
    y_cal_final
)


# ============================================================
# 12. SALVAR MODELO FINAL
# ============================================================

artifact = {
    "features": MODEL_FEATURES,
    "model": modelo_calibrado_final,
    "target": "inundacao_t1",
    "training_cutoff": data_fim_desenvolvimento,
    "forecast_start": FORECAST_START,
    "gap_days": GAP_DAYS,
    "calibration_years": CALIBRATION_YEARS,
}

joblib.dump(
    artifact,
    MODEL_PATH,
)

print("\n============================================")
print("MODELO SALVO")
print("============================================")

print(MODEL_PATH)