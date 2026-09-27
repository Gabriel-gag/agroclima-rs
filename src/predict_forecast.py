from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model_calibrado.joblib"
)


def carregar_modelo():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}"
        )

    modelo = joblib.load(MODEL_PATH)

    if not isinstance(modelo, dict):
        raise TypeError(
            "O model_calibrado.joblib deveria conter "
            "um dicionário."
        )

    if "features" not in modelo:
        raise KeyError(
            "A chave 'features' não existe "
            "no model_calibrado.joblib."
        )

    if "model" not in modelo:
        raise KeyError(
            "A chave 'model' não existe "
            "no model_calibrado.joblib."
        )

    return modelo


def obter_features_modelo(modelo) -> list[str]:

    return list(modelo["features"])


def validar_features(
    df: pd.DataFrame,
    features_modelo: list[str],
) -> None:

    faltantes = [
        coluna
        for coluna in features_modelo
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Features ausentes para previsão:\n"
            + "\n".join(
                f"- {coluna}"
                for coluna in faltantes
            )
        )


def prever_inundacao(
    df_features: pd.DataFrame,
    modelo=None,
) -> pd.DataFrame:

    if modelo is None:
        modelo = carregar_modelo()

    features_modelo = obter_features_modelo(
        modelo
    )

    validar_features(
        df_features,
        features_modelo,
    )

    X = df_features[
        features_modelo
    ].copy()

    clf = modelo["model"]

    probabilidades = (
        clf.predict_proba(X)[:, 1]
    )

    resultado = df_features[
        [
            "date",
            "data_features",
            "station_id",
        ]
    ].copy()

    resultado[
        "probabilidade_inundacao"
    ] = probabilidades

    resultado[
        "probabilidade_pct"
    ] = probabilidades * 100

    # Threshold é mantido apenas como
    # informação complementar.
    threshold = modelo.get(
        "threshold",
        None,
    )

    if threshold is not None:

        resultado[
            "inundacao_predita"
        ] = (
            probabilidades >= threshold
        ).astype(int)

    return resultado


def gerar_previsao(
    df_features: pd.DataFrame,
) -> pd.DataFrame:

    modelo = carregar_modelo()

    return prever_inundacao(
        df_features=df_features,
        modelo=modelo,
    )


if __name__ == "__main__":

    modelo = carregar_modelo()

    print(
        "Modelo calibrado carregado "
        "com sucesso."
    )

    print("\nFeatures utilizadas:")

    for feature in modelo["features"]:
        print(f"- {feature}")

    if "threshold" in modelo:
        print(
            f"\nThreshold: "
            f"{modelo['threshold']}"
        )

    if "target" in modelo:
        print(
            f"Target: "
            f"{modelo['target']}"
        )