from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "data"
    / "forecast"
    / "probabilidades_inundacao_16d.parquet"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "forecast"
    / "ranking_municipios_16d.parquet"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

def carregar_previsoes() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    colunas_obrigatorias = [
        "date",
        "station_id",
        "codigo_ibge",
        "municipio",
        "probabilidade_inundacao",
        "probabilidade_pct",
    ]

    faltantes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Colunas ausentes:\n"
            + "\n".join(
                f"- {coluna}"
                for coluna in faltantes
            )
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["probabilidade_inundacao"] = pd.to_numeric(
        df["probabilidade_inundacao"],
        errors="coerce",
    )

    return df


# ============================================================
# CONSOLIDAR ESTAÇÕES → MUNICÍPIO
# ============================================================

def agregar_municipios(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Remover estações sem código IBGE.
    #
    # Sem código IBGE não conseguimos associar a previsão
    # inequivocamente a um município.
    # --------------------------------------------------------

    df = df[
        df["codigo_ibge"].notna()
    ].copy()

    # --------------------------------------------------------
    # Normalizar código IBGE
    # --------------------------------------------------------

    df["codigo_ibge"] = (
        pd.to_numeric(
            df["codigo_ibge"],
            errors="coerce",
        )
        .astype("Int64")
    )

    df = df[
        df["codigo_ibge"].notna()
    ].copy()

    # --------------------------------------------------------
    # Para cada município e dia:
    #
    # se houver várias estações no mesmo município,
    # usamos a maior probabilidade observada entre elas.
    # --------------------------------------------------------

    df_municipio = (
        df.groupby(
            [
                "date",
                "codigo_ibge",
            ],
            as_index=False,
        )
        .agg(
            probabilidade_inundacao=(
                "probabilidade_inundacao",
                "max",
            )
        )
    )

    # --------------------------------------------------------
    # Recuperar um nome de município por código IBGE.
    #
    # O nome vem da estação, mas o código IBGE é a chave
    # oficial utilizada para consolidar as estações.
    # --------------------------------------------------------

    nomes = (
        df[
            [
                "codigo_ibge",
                "municipio",
            ]
        ]
        .dropna(subset=["municipio"])
        .drop_duplicates(
            subset=["codigo_ibge"]
        )
    )

    df_municipio = df_municipio.merge(
        nomes,
        on="codigo_ibge",
        how="left",
    )

    return df_municipio


# ============================================================
# PROBABILIDADE ACUMULADA
# ============================================================

def calcular_acumulado(
    df_municipio: pd.DataFrame,
) -> pd.DataFrame:

    resultados = []

    for (
        codigo_ibge,
        municipio,
    ), grupo in df_municipio.groupby(
        [
            "codigo_ibge",
            "municipio",
        ]
    ):

        grupo = (
            grupo
            .sort_values("date")
            .reset_index(drop=True)
        )

        probabilidades = (
            grupo[
                "probabilidade_inundacao"
            ]
            .clip(0, 1)
        )

        # Probabilidade de pelo menos uma ocorrência
        # durante o horizonte de previsão.
        prob_acumulada = (
            1
            - np.prod(
                1 - probabilidades
            )
        )

        idx_max = (
            grupo[
                "probabilidade_inundacao"
            ]
            .idxmax()
        )

        linha_max = grupo.loc[
            idx_max
        ]

        resultados.append(
            {
                "codigo_ibge": codigo_ibge,
                "municipio": municipio,
                "probabilidade_acumulada_16d": (
                    prob_acumulada
                ),
                "probabilidade_acumulada_16d_pct": (
                    prob_acumulada * 100
                ),
                "maior_probabilidade_diaria": (
                    linha_max[
                        "probabilidade_inundacao"
                    ]
                ),
                "maior_probabilidade_diaria_pct": (
                    linha_max[
                        "probabilidade_inundacao"
                    ] * 100
                ),
                "data_maior_probabilidade": (
                    linha_max["date"]
                ),
                "dias_disponiveis": len(grupo),
            }
        )

    return pd.DataFrame(resultados)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RANKING DE PROBABILIDADE DE INUNDAÇÃO "
        "— 16 DIAS"
    )
    print("=" * 70)

    
    # --------------------------------------------------------
    # 1. Carregar
    # --------------------------------------------------------

    df = carregar_previsoes()

    print(
        f"\nRegistros carregados: "
        f"{len(df):,}"
    )

    print(
        f"Estações: "
        f"{df['station_id'].nunique()}"
    )

    print(
        f"Municípios: "
        f"{df['codigo_ibge'].nunique()}"
    )

    print(
        f"Período: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # 2. Agregar estações → municípios
    # --------------------------------------------------------

    df_municipio = agregar_municipios(
        df
    )

    print(
        f"\nRegistros municipais: "
        f"{len(df_municipio):,}"
    )
    
    # --------------------------------------------------------
    # 3. Calcular acumulado
    # --------------------------------------------------------

    ranking = calcular_acumulado(
        df_municipio
    )

    print(
    f"Municípios após consolidação: "
    f"{df_municipio['codigo_ibge'].nunique()}"
)

    dias_por_municipio = (
        df_municipio
        .groupby("codigo_ibge")["date"]
        .nunique()
    )

    print(
        f"Municípios com 16 dias completos: "
        f"{(dias_por_municipio == 16).sum()}"
    )

    print(
        f"Municípios com menos de 16 dias: "
        f"{(dias_por_municipio < 16).sum()}"
    )
    # --------------------------------------------------------
    # 4. Ordenar
    # --------------------------------------------------------

    ranking = (
        ranking
        .sort_values(
            "probabilidade_acumulada_16d",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "posicao",
        range(
            1,
            len(ranking) + 1,
        ),
    )

    # --------------------------------------------------------
    # 5. Salvar ranking completo
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Top 5
    # --------------------------------------------------------

    top5 = ranking.head(5).copy()

    print()
    print("=" * 70)
    print("TOP 5 — PROBABILIDADE ACUMULADA EM 16 DIAS")
    print("=" * 70)

    for _, row in top5.iterrows():

        print(
            f"{int(row['posicao'])}. "
            f"{row['municipio']} | "
            f"{row['probabilidade_acumulada_16d_pct']:.4f}% | "
            f"máxima diária: "
            f"{row['maior_probabilidade_diaria_pct']:.4f}% | "
            f"dia: "
            f"{row['data_maior_probabilidade'].date()}"
        )

    # --------------------------------------------------------
    # 7. Resumo
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONCLUÍDO")
    print("=" * 70)

    print(
        f"Municípios avaliados: "
        f"{len(ranking)}"
    )

    print(
        f"Arquivo salvo em:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()