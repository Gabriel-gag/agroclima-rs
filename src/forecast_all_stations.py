from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecast import (
    buscar_previsao,
    agregar_previsao_diaria,
)

from forecast_features import (
    preparar_features_previsao,
)

from predict_forecast import (
    carregar_modelo,
    prever_inundacao,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FORECAST_DAYS = 16
HISTORICAL_DAYS = 7

OUTPUT_DIR = Path("data/forecast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ESTAÇÕES
# ============================================================

STATIONS = {
    # vamos preencher com as 68 estações
}


# ============================================================
# PROCESSAR UMA ESTAÇÃO
# ============================================================

def processar_estacao(
    station_id: str,
    metadata: dict,
    modelo,
) -> pd.DataFrame:

    print()
    print("=" * 60)
    print(f"Estação: {station_id}")
    print(f"Município: {metadata['municipio']}")
    print(f"IBGE: {metadata['codigo_ibge']}")
    print("=" * 60)

    latitude = metadata["latitude"]
    longitude = metadata["longitude"]
    altitude = metadata["altitude"]

    # --------------------------------------------------------
    # 1. Buscar 7 dias de contexto + 16 dias de previsão
    # --------------------------------------------------------

    df_horario = buscar_previsao(
        latitude=latitude,
        longitude=longitude,
        dias=FORECAST_DAYS,
        dias_historico=HISTORICAL_DAYS,
    )

    print(
        f"Dados horários recebidos: "
        f"{len(df_horario)}"
    )

    # --------------------------------------------------------
    # 2. Agregar por dia
    # --------------------------------------------------------

    df_diario = agregar_previsao_diaria(
        df_horario
    )

    print(
        f"Dias disponíveis: "
        f"{len(df_diario)}"
    )

    # --------------------------------------------------------
    # 3. Criar features
    # --------------------------------------------------------

    df_features = preparar_features_previsao(
        df_diario,
        station_id=station_id,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    # --------------------------------------------------------
    # 4. Fazer previsão
    # --------------------------------------------------------

    df_resultado = prever_inundacao(
        df_features,
        modelo=modelo,
    )

    # --------------------------------------------------------
    # 5. Adicionar informações da estação
    # --------------------------------------------------------

    df_resultado["codigo_ibge"] = metadata[
        "codigo_ibge"
    ]

    df_resultado["municipio"] = metadata[
        "municipio"
    ]

    df_resultado["latitude"] = latitude
    df_resultado["longitude"] = longitude
    df_resultado["altitude"] = altitude

    # --------------------------------------------------------
    # 6. Adicionar algumas variáveis meteorológicas
    # --------------------------------------------------------

    colunas_meteorologicas = [
        "precip_mm",
        "precip_sum_24h",
        "precip_sum_48h",
        "precip_sum_72h",
        "precip_sum_7d",
        "rain_days_7d",
        "temperature_avg",
        "humidity_avg",
        "pressure_avg",
        "pressure_change_24h",
        "wind_speed_avg",
    ]

    df_resultado = df_resultado.merge(
        df_features[
            [
                "date",
                "station_id",
                *colunas_meteorologicas,
            ]
        ],
        on=[
            "date",
            "station_id",
        ],
        how="left",
    )

    return df_resultado

def main():

    print("=" * 60)
    print("PREVISÃO PARA TODAS AS ESTAÇÕES")
    print("=" * 60)

    print(
        f"Estações: {len(STATIONS)}"
    )

    # --------------------------------------------------------
    # Carregar modelo apenas uma vez
    # --------------------------------------------------------

    modelo = carregar_modelo()

    resultados = []

    erros = []

    # --------------------------------------------------------
    # Processar estações
    # --------------------------------------------------------

    for i, (station_id, metadata) in enumerate(
        STATIONS.items(),
        start=1,
    ):

        print()
        print(
            f"[{i}/{len(STATIONS)}] "
            f"Processando {station_id}"
        )

        try:

            df_estacao = processar_estacao(
                station_id,
                metadata,
                modelo,
            )

            resultados.append(
                df_estacao
            )

            print(
                f"✅ {station_id} concluída: "
                f"{len(df_estacao)} dias"
            )

        except Exception as exc:

            print(
                f"❌ Erro em {station_id}: "
                f"{exc}"
            )

            erros.append(
                {
                    "station_id": station_id,
                    "erro": str(exc),
                }
            )

    # --------------------------------------------------------
    # Consolidar resultados
    # --------------------------------------------------------

    if resultados:

        df_final = pd.concat(
            resultados,
            ignore_index=True,
        )

        output_path = (
            OUTPUT_DIR
            / "previsoes_68_estacoes.parquet"
        )

        df_final.to_parquet(
            output_path,
            index=False,
        )

        print()
        print("=" * 60)
        print("RESULTADO FINAL")
        print("=" * 60)

        print(
            f"Linhas: {len(df_final)}"
        )

        print(
            f"Estações processadas: "
            f"{df_final['station_id'].nunique()}"
        )

        print(
            f"Municípios: "
            f"{df_final['codigo_ibge'].nunique()}"
        )

        print(
            f"Arquivo: {output_path}"
        )

    # --------------------------------------------------------
    # Salvar erros
    # --------------------------------------------------------

    if erros:

        df_erros = pd.DataFrame(erros)

        erros_path = (
            OUTPUT_DIR
            / "erros_forecast.csv"
        )

        df_erros.to_csv(
            erros_path,
            index=False,
            encoding="utf-8",
        )

        print()
        print(
            f"⚠️ Estações com erro: "
            f"{len(erros)}"
        )

        print(df_erros.to_string(index=False))

    else:

        print()
        print(
            "✅ Todas as estações foram "
            "processadas com sucesso."
        )


if __name__ == "__main__":
    main()