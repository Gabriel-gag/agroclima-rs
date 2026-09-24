import pandas as pd

df_atlas = pd.read_excel("data/atlas/atlas_desastres.xlsx")
df_municipal = pd.read_parquet("data/gold/features_inundacao.parquet")

eventos_atlas = (
    df_atlas[
        (df_atlas["Sigla_UF"].astype(str).str.strip().str.upper() == "RS") &
        (pd.to_numeric(df_atlas["Cod_Cobrade"], errors="coerce") == 12100)
    ]
    .copy()
)

eventos_atlas["Data_Evento"] = pd.to_datetime(
    eventos_atlas["Data_Evento"],
    errors="coerce"
)

eventos_atlas["Cod_IBGE_Mun"] = pd.to_numeric(
    eventos_atlas["Cod_IBGE_Mun"],
    errors="coerce"
)

eventos_atlas = eventos_atlas.drop_duplicates(
    subset=["Cod_IBGE_Mun", "Data_Evento"]
)

eventos_gold = (
    df_municipal[df_municipal["inundacao"] == 1]
    [["codigo_ibge", "date"]]
    .drop_duplicates()
)

print("================================")
print("RESUMO ATLAS x GOLD")
print("================================")

print("Eventos no Atlas:", len(eventos_atlas))
print("Eventos no Gold:", len(eventos_gold))

print("\nMunicípios no Atlas:", eventos_atlas["Cod_IBGE_Mun"].nunique())
print("Municípios no Gold:", eventos_gold["codigo_ibge"].nunique())

print("\nPeríodo Atlas:")
print(
    eventos_atlas["Data_Evento"].min().date(),
    "até",
    eventos_atlas["Data_Evento"].max().date()
)

print("\nEventos Gold:")
print(
    eventos_gold
    .sort_values("date")
    .to_string(index=False)
)