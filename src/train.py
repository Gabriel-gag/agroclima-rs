"""
Treinamento do modelo de probabilidade de inundação - AgroClima RS.

Escopo:
    - Dados meteorológicos de 2025
    - Estações INMET do Rio Grande do Sul
    - Target: inundacao
    - Modelo principal exportado: Random Forest

IMPORTANTE:
    O dataset de 2025 possui poucos eventos positivos de inundação.
    Por isso, este script NÃO realiza uma divisão temporal tradicional
    treino/validação/teste.

    O modelo é treinado utilizando todo o período disponível de 2025.
    As métricas calculadas abaixo são métricas sobre os próprios dados
    utilizados no treinamento e não representam desempenho de
    generalização.

    O objetivo atual é disponibilizar um protótipo de inferência e
    geração de probabilidade para o projeto AgroClima RS.
"""

import os
import joblib

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET = "inundacao"

THRESHOLD = 0.30

YEAR = 2025


# ============================================================
# FEATURES
# ============================================================
#
# Estas são as colunas existentes atualmente no
# data/gold/features_inundacao.parquet.
#
# O train.py e o api.py precisam utilizar exatamente
# os mesmos nomes.
# ============================================================

FEATURE_COLS = [
    "precip_mm",

    "precip_sum_24h",
    "precip_sum_48h",
    "precip_sum_72h",

    "precip_sum_3d",
    "precip_sum_7d",

    "rain_days_7d",
    "dry_days_7d",

    "temperature_min",
    "temperature_max",
    "temperature_avg",
    "temperature_avg_3d",

    "humidity_avg",

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
# XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    HAS_XGB = True

except ImportError:

    HAS_XGB = False


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold=0.50
):
    """
    Calcula métricas de classificação.

    As métricas são calculadas sobre o conjunto recebido.
    Neste projeto, como o modelo é treinado com todo o ano de 2025,
    essas métricas são métricas de treinamento e não de validação.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        )
    }

    # ROC-AUC e PR-AUC precisam das duas classes.
    if pd.Series(y_true).nunique() == 2:

        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities
        )

        result["pr_auc"] = average_precision_score(
            y_true,
            probabilities
        )

    return result


def print_model_metrics(
    model_name,
    metrics
):
    """
    Imprime as métricas de um modelo.
    """

    roc_auc = metrics["roc_auc"]

    pr_auc = metrics["pr_auc"]

    if pd.isna(roc_auc):
        roc_auc_text = "N/A"
    else:
        roc_auc_text = f"{roc_auc:.4f}"

    if pd.isna(pr_auc):
        pr_auc_text = "N/A"
    else:
        pr_auc_text = f"{pr_auc:.4f}"

    print(
        f"{model_name:<22}"
        f"{roc_auc_text:>10}"
        f"{pr_auc_text:>10}"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>10.4f}"
        f"{metrics['f1']:>10.4f}"
    )


def validate_dataset(df):
    """
    Valida o dataset Gold antes do treinamento.
    """

    # --------------------------------------------------------
    # Verificar features
    # --------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLS
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Features ausentes no dataset Gold: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Verificar target
    # --------------------------------------------------------

    if TARGET not in df.columns:

        raise ValueError(
            f"Target '{TARGET}' não encontrado "
            "no dataset Gold."
        )

    # --------------------------------------------------------
    # Converter data
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Converter target
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remover registros inválidos
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            TARGET
        ]
    ).copy()

    # --------------------------------------------------------
    # Garantir target binário
    # --------------------------------------------------------

    target_values = set(
        df[TARGET]
        .astype(int)
        .unique()
    )

    if not target_values.issubset({0, 1}):

        raise ValueError(
            "O target deve conter somente 0 e 1. "
            f"Valores encontrados: {target_values}"
        )

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )

    # --------------------------------------------------------
    # Verificar existência das duas classes
    # --------------------------------------------------------

    if df[TARGET].nunique() < 2:

        raise ValueError(
            "O dataset não possui as duas classes "
            "necessárias para treinamento."
        )

    return df


# ============================================================
# TREINAMENTO
# ============================================================

def train_and_export():

    # ========================================================
    # DIRETÓRIOS
    # ========================================================

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    data_path = os.path.join(
        base_dir,
        "data",
        "gold",
        "features_inundacao.parquet"
    )

    output_model_path = os.path.join(
        os.path.dirname(__file__),
        "model.joblib"
    )

    # ========================================================
    # VERIFICAR DATASET
    # ========================================================

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            "Dataset Gold não encontrado:\n"
            f"{data_path}"
        )

    # ========================================================
    # CARREGAR DATASET
    # ========================================================

    print("=" * 75)

    print(
        "CARREGANDO DATASET GOLD"
    )

    print("=" * 75)

    df = pd.read_parquet(
        data_path
    )

    # ========================================================
    # VALIDAR DATASET
    # ========================================================

    df = validate_dataset(
        df
    )

    # ========================================================
    # FILTRAR 2025
    # ========================================================

    df = df[
        df["date"].dt.year == YEAR
    ].copy()

    # ========================================================
    # VERIFICAR SE EXISTEM DADOS
    # ========================================================

    if df.empty:

        raise ValueError(
            f"Nenhum registro encontrado para {YEAR}."
        )

    # ========================================================
    # ORDENAR
    # ========================================================

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ========================================================
    # INFORMAÇÕES DO DATASET
    # ========================================================

    print(
        f"\nPeríodo utilizado:"
        f" {df['date'].min().date()}"
        f" -> "
        f"{df['date'].max().date()}"
    )

    print(
        f"Registros: "
        f"{len(df):,}"
    )

    print(
        "\nDistribuição do target:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    print(
        "\nProporção:"
    )

    print(
        df[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    positive_count = int(
        (
            df[TARGET] == 1
        ).sum()
    )

    negative_count = int(
        (
            df[TARGET] == 0
        ).sum()
    )

    print(
        f"\nEventos positivos: "
        f"{positive_count}"
    )

    print(
        f"Não-eventos: "
        f"{negative_count}"
    )

    # ========================================================
    # LISTAR EVENTOS POSITIVOS
    # ========================================================

    positives = df[
        df[TARGET] == 1
    ][
        [
            "station_id",
            "municipio",
            "date"
        ]
    ].copy()

    print(
        "\nEventos de inundação utilizados:"
    )

    if positives.empty:

        print(
            "Nenhum evento positivo encontrado."
        )

    else:

        print(
            positives
            .sort_values("date")
            .to_string(
                index=False
            )
        )

    # ========================================================
    # AVISO METODOLÓGICO
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AVISO METODOLÓGICO"
    )

    print(
        "=" * 75
    )

    print(
        "O modelo será treinado utilizando todo o período de 2025."
    )

    print(
        "Não será realizado split treino/validação/teste,"
    )

    print(
        "pois existem poucos eventos positivos e eles estão"
    )

    print(
        "concentrados em 2025."
    )

    print(
        "\nAs métricas apresentadas abaixo são métricas"
    )

    print(
        "sobre os próprios dados de treinamento."
    )

    print(
        "Elas NÃO representam desempenho em dados futuros."
    )

    print(
        "=" * 75
    )

    # ========================================================
    # SEPARAR X E Y
    # ========================================================

    X = df[
        FEATURE_COLS
    ].copy()

    y = df[
        TARGET
    ].copy()

    # ========================================================
    # IMPUTAÇÃO
    # ========================================================

    print(
        "\nAplicando imputação..."
    )

    imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)
    X_imputed = (
        imputer
        .fit_transform(X)
    )
    print(
    f"Features antes da imputação: {X.shape[1]}"
)

    print(
        f"Features após a imputação: {X_imputed.shape[1]}"
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    print(
        "Treinando Logistic Regression..."
    )

    scaler = StandardScaler()

    X_scaled = (
        scaler
        .fit_transform(
            X_imputed
        )
    )

    logistic = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    )

    logistic.fit(
        X_scaled,
        y
    )

    probabilities_lr = (
        logistic
        .predict_proba(
            X_scaled
        )[:, 1]
    )

    metrics_lr = evaluate_model(
        y,
        probabilities_lr,
        THRESHOLD
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "Treinando Random Forest..."
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_imputed,
        y
    )

    probabilities_rf = (
        random_forest
        .predict_proba(
            X_imputed
        )[:, 1]
    )

    metrics_rf = evaluate_model(
        y,
        probabilities_rf,
        THRESHOLD
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    xgboost_model = None

    metrics_xgb = None

    if HAS_XGB:

        print(
            "Treinando XGBoost..."
        )

        xgboost_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        xgboost_model.fit(
            X_imputed,
            y
        )

        probabilities_xgb = (
            xgboost_model
            .predict_proba(
                X_imputed
            )[:, 1]
        )

        metrics_xgb = evaluate_model(
            y,
            probabilities_xgb,
            THRESHOLD
        )

    else:

        print(
            "XGBoost não instalado."
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MÉTRICAS SOBRE O TREINAMENTO"
    )

    print(
        "=" * 75
    )

    print(
        f"\n"
        f"{'Modelo':<22}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print(
        "-" * 75
    )

    print_model_metrics(
        "Logistic Regression",
        metrics_lr
    )

    print_model_metrics(
        "Random Forest",
        metrics_rf
    )

    if metrics_xgb is not None:

        print_model_metrics(
            "XGBoost",
            metrics_xgb
        )

    else:

        print(
            f"{'XGBoost':<22}"
            "indisponível"
        )

    print(
        "=" * 75
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    print(
        f"\nThreshold utilizado: "
        f"{THRESHOLD:.2f}"
    )

    print(
        "\nO threshold não foi otimizado "
        "sobre um conjunto de validação."
    )

    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    print(
        "\nMatriz de confusão - Random Forest:"
    )

    print(
        metrics_rf[
            "confusion_matrix"
        ]
    )

    # ========================================================
    # IMPORTÂNCIA DAS FEATURES
    # ========================================================
    print("\nQuantidade de features:")
    print("FEATURE_COLS:", len(FEATURE_COLS))
    print("X:", X.shape[1])
    print("Random Forest:", len(random_forest.feature_importances_))
    # ========================================================
# IMPORTÂNCIA DAS FEATURES
# ========================================================

    importances = random_forest.feature_importances_

    if len(FEATURE_COLS) != len(importances):

        raise ValueError(
            "\nInconsistência entre as features e o modelo:\n"
            f"FEATURE_COLS: {len(FEATURE_COLS)}\n"
            f"X: {X.shape[1]}\n"
            f"Random Forest: {len(importances)}\n"
        )

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": importances
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nTop 10 features - Random Forest:"
    )

    print(
        feature_importance
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # ARTEFATO
    # ========================================================

    artifact = {

        "model": random_forest,

        "imputer": imputer,

        "features": FEATURE_COLS,

        "threshold": THRESHOLD,

        "target": TARGET,

        "model_type":
            "RandomForestClassifier",

        "training_period": {

            "start":
                df["date"]
                .min()
                .strftime(
                    "%Y-%m-%d"
                ),

            "end":
                df["date"]
                .max()
                .strftime(
                    "%Y-%m-%d"
                )
        },

        "training_year":
            YEAR,

        "training_rows":
            len(df),

        "positive_events":
            positive_count,

        "negative_events":
            negative_count,

        "evaluation_type":
            "training_only",

        "evaluation_note":
            (
                "Modelo treinado com todos os "
                "dados disponíveis de 2025. "
                "As métricas não representam "
                "desempenho de generalização."
            ),

        "metrics_training": {

            "roc_auc":
                metrics_rf["roc_auc"],

            "pr_auc":
                metrics_rf["pr_auc"],

            "precision":
                metrics_rf["precision"],

            "recall":
                metrics_rf["recall"],

            "f1":
                metrics_rf["f1"]
        }
    }

    # ========================================================
    # SALVAR MODELO
    # ========================================================

    joblib.dump(
        artifact,
        output_model_path
    )

    print(
        "\nModelo exportado com sucesso:"
    )

    print(
        output_model_path
    )

    # ========================================================
    # CONFIRMAÇÃO DAS FEATURES
    # ========================================================

    print(
        "\nFeatures armazenadas no model.joblib:"
    )

    for feature in FEATURE_COLS:

        print(
            f"  - {feature}"
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TREINAMENTO CONCLUÍDO"
    )

    print(
        "=" * 75
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    train_and_export()