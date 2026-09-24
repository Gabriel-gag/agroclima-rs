import pandas as pd
import geopandas as gpd

# Gold
df_gold = pd.read_parquet("data/gold/features_inundacao.parquet")

# GeoJSON dos municípios do RS
gdf = gpd.read_file("data/geo/rs_municipios.geojson")

# Normaliza tipos
gold_codes = (
    pd.to_numeric(df_gold["codigo_ibge"], errors="coerce")
    .dropna()
    .astype("int64")
    .unique()
)

geo_codes = (
    pd.to_numeric(gdf["CD_MUN"], errors="coerce")
    .dropna()
    .astype("int64")
    .unique()
)

# Códigos do Gold que não existem no GeoJSON
invalidos = sorted(set(gold_codes) - set(geo_codes))

print("Códigos IBGE no Gold:", len(gold_codes))
print("Códigos IBGE no GeoJSON:", len(geo_codes))
print("Códigos inválidos:", len(invalidos))

if invalidos:
    print(invalidos)
else:
    print("✅ Todos os códigos IBGE do Gold existem no GeoJSON.")

    estacoes = (
    df_gold[
        ["station_id", "municipio", "codigo_ibge"]
    ]
    .drop_duplicates()
    .sort_values("station_id")
)

print(estacoes.to_string(index=False))