from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd


# ============================================================
# CONFIGURAÇÃO DE CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

GOLD_PATH = (
    ROOT_DIR
    / "data"
    / "gold"
    / "features_inundacao.parquet"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "forecast"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "probabilidades_inundacao_16d.parquet"
)


# ============================================================
# IMPORTS DO PROJETO
# ============================================================

sys.path.insert(
    0,
    str(ROOT_DIR)
)

from src.forecast import (
    buscar_previsao,
    agregar_previsao_diaria,
)

from src.forecast_features import (
    preparar_features_previsao,
)

from src.predict_forecast import (
    gerar_previsao,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

FORECAST_DAYS = 16
HISTORICAL_DAYS = 7

# Pequena pausa entre estações para evitar
# bombardear a API.
SLEEP_BETWEEN_STATIONS = 0.5


# ============================================================
# LEITURA DAS ESTAÇÕES
# ============================================================

def carregar_estacoes() -> pd.DataFrame:

    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold não encontrado em:\n{GOLD_PATH}"
        )

    print(
        f"📂 Carregando Gold:\n{GOLD_PATH}"
    )

    df = pd.read_parquet(
        GOLD_PATH
    )

    colunas_obrigatorias = [
        "station_id",
        "latitude",
        "longitude",
        "altitude",
        "codigo_ibge",
        "municipio",
    ]

    faltantes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Colunas necessárias para identificar "
            "as estações estão ausentes no Gold:\n"
            + "\n".join(
                f"- {coluna}"
                for coluna in faltantes
            )
        )

    estacoes = (
        df[
            colunas_obrigatorias
        ]
        .drop_duplicates(
            subset=["station_id"]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Conversão numérica
    # --------------------------------------------------------

    for coluna in [
        "latitude",
        "longitude",
        "altitude",
    ]:
        estacoes[coluna] = pd.to_numeric(
            estacoes[coluna],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remover estações sem coordenadas
    # --------------------------------------------------------

    antes = len(estacoes)

    estacoes = estacoes[
        estacoes["latitude"].notna()
        & estacoes["longitude"].notna()
    ].copy()

    removidas = antes - len(estacoes)

    if removidas:
        print(
            f"⚠️ {removidas} estação(ões) "
            "sem coordenadas foram ignoradas."
        )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    estacoes = (
        estacoes
        .sort_values("station_id")
        .reset_index(drop=True)
    )

    return estacoes


# ============================================================
# PROCESSAMENTO DE UMA ESTAÇÃO
# ============================================================

def processar_estacao(
    estacao: pd.Series,
) -> pd.DataFrame:

    station_id = str(
        estacao["station_id"]
    )

    latitude = float(
        estacao["latitude"]
    )

    longitude = float(
        estacao["longitude"]
    )

    altitude = float(
        estacao["altitude"]
    )

    # --------------------------------------------------------
    # 1. Buscar Open-Meteo
    # --------------------------------------------------------

    df_horario = buscar_previsao(
        latitude=latitude,
        longitude=longitude,
        dias=FORECAST_DAYS,
        dias_historico=HISTORICAL_DAYS,
    )

    # --------------------------------------------------------
    # 2. Agregar para escala diária
    # --------------------------------------------------------

    df_diario = agregar_previsao_diaria(
        df_horario
    )

    # --------------------------------------------------------
    # 3. Criar features D+1 ... D+16
    # --------------------------------------------------------

    df_features = preparar_features_previsao(
        df_diario,
        station_id=station_id,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    # --------------------------------------------------------
    # 4. Aplicar modelo calibrado
    # --------------------------------------------------------

    df_pred = gerar_previsao(
        df_features
    )

    # --------------------------------------------------------
    # 5. Adicionar informações da estação
    # --------------------------------------------------------

    df_pred["codigo_ibge"] = (
        estacao["codigo_ibge"]
    )

    df_pred["municipio"] = (
        estacao["municipio"]
    )

    df_pred["latitude"] = latitude
    df_pred["longitude"] = longitude
    df_pred["altitude"] = altitude

    return df_pred


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    inicio = time.time()

    print("=" * 70)
    print(
        "GERAÇÃO DE PREVISÃO DE INUNDAÇÃO — "
        "RIO GRANDE DO SUL"
    )
    print("=" * 70)

    print(
        f"\nHorizonte: {FORECAST_DAYS} dias"
    )

    print(
        f"Contexto histórico: "
        f"{HISTORICAL_DAYS} dias"
    )

    # --------------------------------------------------------
    # Criar diretório de saída
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Carregar estações
    # --------------------------------------------------------

    estacoes = carregar_estacoes()

    total_estacoes = len(
        estacoes
    )

    print(
        f"\n📡 Estações encontradas: "
        f"{total_estacoes}"
    )

    print(
        f"📅 Previsões esperadas: "
        f"{total_estacoes * FORECAST_DAYS}"
    )

    print()

    # --------------------------------------------------------
    # Processamento
    # --------------------------------------------------------

    resultados = []

    sucessos = 0
    erros = 0

    for indice, (_, estacao) in enumerate(
        estacoes.iterrows(),
        start=1,
    ):

        station_id = str(
            estacao["station_id"]
        )

        municipio = str(
            estacao["municipio"]
        )

        print(
            f"[{indice}/{total_estacoes}] "
            f"{station_id} — {municipio}"
        )

        try:

            resultado = processar_estacao(
                estacao
            )

            # ------------------------------------------------
            # Validação básica
            # ------------------------------------------------

            if len(resultado) != FORECAST_DAYS:
                raise ValueError(
                    f"Esperadas "
                    f"{FORECAST_DAYS} previsões, "
                    f"mas foram obtidas "
                    f"{len(resultado)}."
                )

            if resultado[
                "probabilidade_inundacao"
            ].isna().any():

                raise ValueError(
                    "Foram encontradas "
                    "probabilidades NaN."
                )

            # ------------------------------------------------
            # Guardar
            # ------------------------------------------------

            resultados.append(
                resultado
            )

            sucessos += 1

            prob_max = (
                resultado[
                    "probabilidade_pct"
                ]
                .max()
            )

            print(
                f"   ✓ {len(resultado)} dias"
                f" | máxima: "
                f"{prob_max:.4f}%"
            )

        except Exception as exc:

            erros += 1

            print(
                f" Erro: {exc}"
            )

        # ----------------------------------------------------
        # Pequena pausa entre chamadas
        # ----------------------------------------------------

        if (
            indice < total_estacoes
            and SLEEP_BETWEEN_STATIONS > 0
        ):
            time.sleep(
                SLEEP_BETWEEN_STATIONS
            )

    # ========================================================
    # CONSOLIDAÇÃO
    # ========================================================

    print()
    print("=" * 70)
    print("CONSOLIDAÇÃO")
    print("=" * 70)

    if not resultados:

        raise RuntimeError(
            "Nenhuma estação foi processada "
            "com sucesso."
        )

    df_final = pd.concat(
        resultados,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    df_final = (
        df_final
        .sort_values(
            [
                "date",
                "station_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Garantir tipos
    # --------------------------------------------------------

    df_final["date"] = pd.to_datetime(
        df_final["date"]
    )

    df_final["data_features"] = pd.to_datetime(
        df_final["data_features"]
    )

    # --------------------------------------------------------
    # Validações finais
    # --------------------------------------------------------

    print(
        f"\nEstações processadas: "
        f"{sucessos}/{total_estacoes}"
    )

    print(
        f"Estações com erro: "
        f"{erros}"
    )

    print(
        f"Total de registros: "
        f"{len(df_final):,}"
    )

    registros_esperados = (
        sucessos * FORECAST_DAYS
    )

    print(
        f"Registros esperados: "
        f"{registros_esperados:,}"
    )

    if len(df_final) != registros_esperados:

        raise ValueError(
            "Quantidade final de registros "
            "não corresponde ao esperado."
        )

    # --------------------------------------------------------
    # Verificar duplicidades
    # --------------------------------------------------------

    duplicados = df_final.duplicated(
        subset=[
            "station_id",
            "date",
        ]
    ).sum()

    if duplicados > 0:

        raise ValueError(
            f"Foram encontradas "
            f"{duplicados} duplicidades "
            "station_id + date."
        )

    # --------------------------------------------------------
    # Verificar intervalo de probabilidades
    # --------------------------------------------------------

    prob = df_final[
        "probabilidade_inundacao"
    ]

    if (
        (prob < 0).any()
        or (prob > 1).any()
    ):

        raise ValueError(
            "Existem probabilidades fora "
            "do intervalo [0, 1]."
        )

    # ========================================================
    # SALVAR
    # ========================================================

    print(
        f"\n💾 Salvando:"
    )

    print(
        OUTPUT_PATH
    )

    df_final.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # RESUMO
    # ========================================================

    data_min = (
        df_final["date"].min()
    )

    data_max = (
        df_final["date"].max()
    )

    print()
    print("=" * 70)
    print("CONCLUÍDO")
    print("=" * 70)

    print(
        f"Período previsto: "
        f"{data_min.date()} → "
        f"{data_max.date()}"
    )

    print(
        f"Estações: "
        f"{df_final['station_id'].nunique()}"
    )

    print(
        f"Municípios: "
        f"{df_final['codigo_ibge'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(df_final):,}"
    )

    print(
        f"Arquivo: "
        f"{OUTPUT_PATH}"
    )

    tempo = (
        time.time() - inicio
    )

    print(
        f"Tempo total: "
        f"{tempo / 60:.2f} minutos"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()