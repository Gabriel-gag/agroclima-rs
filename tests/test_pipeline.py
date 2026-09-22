import os
import unittest

import pandas as pd

import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../src"
    )
)

from transformation import (
    transform_bronze_to_silver,
    transform_silver_to_gold
)

from api import app

from fastapi.testclient import TestClient


class TestAgroClimaPipeline(
    unittest.TestCase
):

    def setUp(self):

        base_dir = os.path.join(
            os.path.dirname(__file__),
            ".."
        )

        self.bronze_dir = os.path.join(
            base_dir,
            "data",
            "bronze"
        )

        self.atlas_path = os.path.join(
            base_dir,
            "data",
            "atlas",
            "atlas_desastres.xlsx"
        )

        self.silver_path = (
            "/tmp/test_fact_weather_daily.parquet"
        )

        self.gold_path = (
            "/tmp/test_features_inundacao.parquet"
        )

    def tearDown(self):

        paths = [
            self.silver_path,
            self.silver_path.replace(
                ".parquet",
                ".csv"
            ),
            self.gold_path,
            self.gold_path.replace(
                ".parquet",
                ".csv"
            )
        ]

        for path in paths:

            if os.path.exists(path):
                os.remove(path)

    # ========================================================
    # SILVER
    # ========================================================

    def test_transformation_bronze_to_silver_real(
        self
    ):
        """
        Verifica se todas as estações RS presentes
        no Bronze conseguem ser transformadas em
        registros diários.
        """

        df_silver = transform_bronze_to_silver(
            bronze_dir=self.bronze_dir,
            output_path=self.silver_path
        )

        self.assertGreater(
            len(df_silver),
            0
        )

        self.assertIn(
            "station_id",
            df_silver.columns
        )

        self.assertIn(
            "municipio",
            df_silver.columns
        )

        self.assertIn(
            "codigo_ibge",
            df_silver.columns
        )

        self.assertIn(
            "date",
            df_silver.columns
        )

        # ----------------------------------------------------
        # Todas as estações devem possuir dados
        # ----------------------------------------------------

        self.assertGreater(
            df_silver[
                "station_id"
            ].nunique(),
            1
        )

        # ----------------------------------------------------
        # codigo_ibge pode ser nulo.
        #
        # O catálogo do projeto não exige que cada estação
        # tenha um código IBGE diretamente cadastrado.
        # O código será preenchido quando houver
        # correspondência com o Atlas.
        # ----------------------------------------------------

        invalid_ibge = df_silver[
            df_silver["codigo_ibge"].notna()
            &
            (
                pd.to_numeric(
                    df_silver["codigo_ibge"],
                    errors="coerce"
                ) <= 0
            )
        ]

        self.assertEqual(
            len(invalid_ibge),
            0
        )

    # ========================================================
    # GOLD
    # ========================================================

    def test_transformation_silver_to_gold_real(
        self
    ):
        """
        Verifica a criação das features e do
        target de inundação.
        """

        df_silver = transform_bronze_to_silver(
            bronze_dir=self.bronze_dir,
            output_path=self.silver_path
        )

        df_gold = transform_silver_to_gold(
            silver_path=self.silver_path,
            atlas_path=self.atlas_path,
            output_path=self.gold_path
        )

        self.assertIn(
            "inundacao",
            df_gold.columns
        )

        self.assertIn(
            "pressure_change_24h",
            df_gold.columns
        )

        self.assertIn(
            "precip_sum_7d",
            df_gold.columns
        )

        self.assertIn(
            "precip_sum_72h",
            df_gold.columns
        )

        # Target binário
        targets = set(
            df_gold[
                "inundacao"
            ]
            .dropna()
            .unique()
        )

        self.assertTrue(
            targets.issubset(
                {0, 1}
            )
        )

        # Deve existir pelo menos algum negativo
        self.assertIn(
            0,
            targets
        )

    # ========================================================
    # API
    # ========================================================

    def test_api_live_inference(
        self
    ):
        """
        Testa o endpoint de probabilidade
        de inundação.
        """

        client = TestClient(
            app,
            raise_server_exceptions=False
)

        res = client.get(
            "/inundacao?station_id=A887"
        )

        print(
            "\n=== RESPOSTA DA API ==="
        )

        print(
            "Status:",
            res.status_code
        )

        print(
            "Body:",
            res.text
        )

        self.assertEqual(
            res.status_code,
            200
        )

        data = res.json()

        self.assertIn(
            "inferencia_modelo",
            data
        )

        self.assertIn(
            "probabilidade_inundacao",
            data["inferencia_modelo"]
        )
        self.assertIn(
            "probabilidade_percentual",
            data["inferencia_modelo"]
        )

        self.assertIn(
            "inundacao_prevista",
            data["inferencia_modelo"]
        )

        self.assertIsInstance(
            data["inferencia_modelo"]["inundacao_prevista"],
            bool
        )

        probability = data[
            "inferencia_modelo"
        ]["probabilidade_inundacao"]

        self.assertGreaterEqual(
            probability,
            0
        )

        self.assertLessEqual(
            probability,
            1
        )

        self.assertGreaterEqual(
            probability,
            0
        )

        self.assertLessEqual(
            probability,
            1
        )


if __name__ == "__main__":

    unittest.main()