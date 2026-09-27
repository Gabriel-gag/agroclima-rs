from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import time

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

FORECAST_DAYS = 16

CACHE_DIR = Path("data/forecast")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]


def buscar_previsao(
    latitude: float,
    longitude: float,
    dias: int = FORECAST_DAYS,
    dias_historico: int = 7,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:

    if not 1 <= dias <= 16:
        raise ValueError("O número de dias de previsão deve estar entre 1 e 16.")

    if not 1 <= dias_historico <= 14:
        raise ValueError("O histórico recente deve estar entre 1 e 14 dias.")

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": ",".join(HOURLY_VARIABLES),

        "past_days": dias_historico,
        "forecast_days": dias,

        "timezone": timezone,

        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "ms",
    }

    ultima_excecao = None

    for tentativa in range(3):
        try:
            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=(30, 120),
            )

            response.raise_for_status()
            break

        except requests.exceptions.RequestException as exc:
            ultima_excecao = exc

            if tentativa < 2:
                espera = 5 * (tentativa + 1)

                print(
                    f"⚠️ Falha na tentativa "
                    f"{tentativa + 1}/3. "
                    f"Tentando novamente em {espera}s..."
                )

                time.sleep(espera)

            else:
                raise ultima_excecao

    dados = response.json()

    if "hourly" not in dados:
        raise ValueError(
            "Resposta do Open-Meteo não contém dados horários."
        )

    df = pd.DataFrame(dados["hourly"])

    df["datetime"] = pd.to_datetime(df["time"])

    df.drop(columns=["time"], inplace=True)

    return df


def agregar_previsao_diaria(
    df_horario: pd.DataFrame,
) -> pd.DataFrame:

    df = df_horario.copy()

    df["date"] = df["datetime"].dt.normalize()

    df_diario = (
        df.groupby("date")
        .agg(
            precip_mm=("precipitation", "sum"),
            precip_probability_max=(
                "precipitation_probability",
                "max",
            ),
            temperature_min=(
                "temperature_2m",
                "min",
            ),
            temperature_max=(
                "temperature_2m",
                "max",
            ),
            temperature_avg=(
                "temperature_2m",
                "mean",
            ),
            humidity_avg=(
                "relative_humidity_2m",
                "mean",
            ),
            pressure_avg=(
                "surface_pressure",
                "mean",
            ),
            wind_speed_avg=(
                "wind_speed_10m",
                "mean",
            ),
            wind_gust_max=(
                "wind_gusts_10m",
                "max",
            ),
            solar_radiation=(
                "shortwave_radiation",
                "sum",
            ),
        )
        .reset_index()
    )

    return df_diario