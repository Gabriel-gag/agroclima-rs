from __future__ import annotations

import numpy as np
import pandas as pd


HISTORICAL_DAYS = 7


BASE_WEATHER_COLUMNS = [
    "precip_mm",
    "temperature_min",
    "temperature_max",
    "temperature_avg",
    "humidity_avg",
    "pressure_avg",
    "wind_speed_avg",
    "solar_radiation",
]


def preparar_contexto_previsao(
    df_previsao: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude: float,
    station_id: str,
    dias_historico: int = HISTORICAL_DAYS,
) -> pd.DataFrame:

    df = df_previsao.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    if df["date"].isna().any():
        raise ValueError(
            "Existem datas inválidas nos dados de previsão."
        )

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    if len(df) <= dias_historico:
        raise ValueError(
            "Dados insuficientes para separar "
            "histórico e previsão."
        )

    # Os primeiros dias serão utilizados como
    # contexto histórico.
    datas = df["date"].sort_values().unique()

    data_inicio_forecast = pd.Timestamp(
        datas[dias_historico]
    )

    df["origem"] = np.where(
        df["date"] < data_inicio_forecast,
        "historico",
        "forecast",
    )

    # Metadados da estação
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["altitude"] = altitude
    df["station_id"] = station_id

    return df


def criar_features_forecast(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        df.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ==========================================
    # PRECIPITAÇÃO
    # ==========================================

    df["precip_sum_24h"] = (
        df["precip_mm"]
        .rolling(
            window=1,
            min_periods=1,
        )
        .sum()
    )

    df["precip_sum_48h"] = (
        df["precip_mm"]
        .rolling(
            window=2,
            min_periods=2,
        )
        .sum()
    )

    df["precip_sum_72h"] = (
        df["precip_mm"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    df["precip_sum_3d"] = (
        df["precip_sum_72h"]
    )

    df["precip_sum_7d"] = (
        df["precip_mm"]
        .rolling(
            window=7,
            min_periods=7,
        )
        .sum()
    )

    # ==========================================
    # DIAS DE CHUVA / SECA
    # ==========================================

    df["rain_days_7d"] = (
        df["precip_mm"]
        .ge(1)
        .rolling(
            window=7,
            min_periods=7,
        )
        .sum()
    )

    df["dry_days_7d"] = (
        df["precip_mm"]
        .lt(1)
        .rolling(
            window=7,
            min_periods=7,
        )
        .sum()
    )

    # ==========================================
    # TEMPERATURA
    # ==========================================

    df["temperature_avg_3d"] = (
        df["temperature_avg"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .mean()
    )

    # ==========================================
    # UMIDADE
    # ==========================================

    df["humidity_avg_3d"] = (
        df["humidity_avg"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .mean()
    )

    # ==========================================
    # PRESSÃO
    # ==========================================

    df["pressure_change_24h"] = (
        df["pressure_avg"].diff(1)
    )

    # ==========================================
    # SAZONALIDADE
    # ==========================================

    day_of_year = df["date"].dt.dayofyear

    df["season_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["season_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return df


def preparar_features_previsao(
    df_previsao_diaria: pd.DataFrame,
    station_id: str,
    latitude: float,
    longitude: float,
    altitude: float,
) -> pd.DataFrame:

    # ------------------------------------------
    # 1. Criar contexto histórico + previsão
    # ------------------------------------------

    df = preparar_contexto_previsao(
        df_previsao_diaria,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        station_id=station_id,
        dias_historico=HISTORICAL_DAYS,
    )

    # ------------------------------------------
    # 2. Criar features sobre todo o período
    # ------------------------------------------

    df = criar_features_forecast(df)

    # ------------------------------------------
    # 3. Separar as linhas que servirão como
    #    entrada do modelo
    #
    #    T     -> prevê T+1
    #    T+1   -> prevê T+2
    #    ...
    # ------------------------------------------

    forecast_dates = (
        df.loc[df["origem"] == "forecast", "date"]
        .sort_values()
        .reset_index(drop=True)
    )

    data_inicio_forecast = forecast_dates.iloc[0]

    # A primeira previsão (D+1) usa o último
    # dia do histórico como entrada.
    #
    # Portanto precisamos das linhas:
    #
    # último dia histórico
    # + 15 primeiros dias do forecast
    #
    # total = 16 linhas

    data_inicio_features = (
        data_inicio_forecast
        - pd.Timedelta(days=1)
    )

    data_fim_features = (
        forecast_dates.iloc[-1]
        - pd.Timedelta(days=1)
    )

    resultado = df[
        (df["date"] >= data_inicio_features)
        & (df["date"] <= data_fim_features)
    ].copy()

    resultado = (
        resultado
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Guarda a data em que as features estão
    # disponíveis.
    resultado["data_features"] = resultado["date"]

    # A data prevista é sempre o dia seguinte.
    resultado["date"] = (
        resultado["data_features"]
        + pd.Timedelta(days=1)
    )

    return resultado