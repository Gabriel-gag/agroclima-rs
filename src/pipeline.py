"""
Orquestrador Principal do Pipeline AgroClima RS

Bronze -> Silver -> Gold
"""

import os
import sys

sys.path.append(
    os.path.dirname(__file__)
)

from ingestion import (
    ingest_from_inmet_zip
)

from transformation import (
    transform_bronze_to_silver,
    transform_silver_to_gold
)


def run_pipeline(
    zip_path=None
):

    print("=" * 70)
    print(
        "INICIANDO PIPELINE AGROCLIMA RS"
    )
    print(
        "Escopo: todas as estações do Rio Grande do Sul"
    )
    print("=" * 70)

    base_dir = os.path.join(
        os.path.dirname(__file__),
        ".."
    )

    bronze_dir = os.path.join(
        base_dir,
        "data",
        "bronze"
    )

    silver_path = os.path.join(
        base_dir,
        "data",
        "silver",
        "fact_weather_daily.parquet"
    )

    gold_path = os.path.join(
        base_dir,
        "data",
        "gold",
        "features_inundacao.parquet"
    )

    atlas_path = os.path.join(
        base_dir,
        "data",
        "atlas",
        "atlas_desastres.xlsx"
    )

    # ========================================================
    # BRONZE
    # ========================================================

    if zip_path is not None:

        if not os.path.exists(zip_path):

            raise FileNotFoundError(
                f"ZIP não encontrado: {zip_path}"
            )

        print(
            f"\n1. Bronze"
        )

        print(
            f"   Extraindo: {zip_path}"
        )

        extracted = ingest_from_inmet_zip(
            zip_path,
            output_dir=bronze_dir
        )

        print(
            f"   -> {len(extracted)} "
            f"arquivos extraídos"
        )

    else:

        csv_files = [
            f
            for f in os.listdir(
                bronze_dir
            )
            if f.lower().endswith(".csv")
        ] if os.path.exists(
            bronze_dir
        ) else []

        if not csv_files:

            raise FileNotFoundError(
                "data/bronze não possui "
                "arquivos CSV de estações."
            )

        print(
            "\n1. Bronze"
        )

        print(
            f"   -> Utilizando {len(csv_files)} "
            f"arquivos locais"
        )

    # ========================================================
    # SILVER
    # ========================================================

    print(
        "\n2. Silver"
    )

    df_silver = transform_bronze_to_silver(
        bronze_dir=bronze_dir,
        output_path=silver_path
    )

    print(
        f"   -> {len(df_silver)} "
        f"registros diários"
    )

    print(
        f"   -> "
        f"{df_silver['station_id'].nunique()} "
        f"estações"
    )

    # ========================================================
    # GOLD
    # ========================================================

    print(
        "\n3. Gold"
    )

    df_gold = transform_silver_to_gold(
        silver_path=silver_path,
        atlas_path=atlas_path,
        output_path=gold_path
    )

    print(
        f"   -> {len(df_gold)} "
        f"registros para modelagem"
    )

    print(
        f"   -> "
        f"{df_gold['station_id'].nunique()} "
        f"estações"
    )

    print(
        f"   -> "
        f"{df_gold['inundacao'].sum()} "
        f"observações positivas"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE EXECUTADO COM SUCESSO"
    )

    print(
        "Dados disponíveis em:"
    )

    print(
        "  data/bronze/"
    )

    print(
        "  data/silver/"
    )

    print(
        "  data/gold/"
    )

    print(
        "=" * 70
    )

    return df_gold


if __name__ == "__main__":

    zip_arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    run_pipeline(
        zip_path=zip_arg
    )