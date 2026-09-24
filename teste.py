import pandas as pd

df_gold = pd.read_parquet(
    "data/gold/features_inundacao.parquet"
)

cobertura = (
    df_gold
    .groupby(["codigo_ibge", "station_id"])
    .agg(
        data_inicio=("date", "min"),
        data_fim=("date", "max"),
        dias=("date", "nunique"),
        linhas=("date", "size")
    )
    .reset_index()
    .sort_values(["codigo_ibge", "station_id"])
)

print(cobertura.to_string(index=False))

cobertura = (
    df_gold
    .groupby(["codigo_ibge", "station_id"])
    .agg(
        data_inicio=("date", "min"),
        data_fim=("date", "max"),
        dias=("date", "nunique"),
        linhas=("date", "size")
    )
    .reset_index()
    .sort_values(["codigo_ibge", "station_id"])
)


# Seleciona a estação com maior cobertura para cada município
estacao_representativa = (
    cobertura
    .sort_values(
        ["codigo_ibge", "dias", "station_id"],
        ascending=[True, False, True]
    )
    .drop_duplicates("codigo_ibge")
    .reset_index(drop=True)
)

print(estacao_representativa.to_string(index=False))



codigos_teste = [
    4317301,  # Santa Vitória do Palmar
    4301602,  # Bagé
    4305108,  # Caxias do Sul
    4314902   # Porto Alegre
]

print(
    estacao_representativa[
        estacao_representativa["codigo_ibge"].isin(codigos_teste)
    ].to_string(index=False)
)

# Seleciona apenas as estações representativas
df_municipal = df_gold.merge(
    estacao_representativa[
        ["codigo_ibge", "station_id"]
    ],
    on=["codigo_ibge", "station_id"],
    how="inner"
)

print("Linhas:", len(df_municipal))
print("Municípios:", df_municipal["codigo_ibge"].nunique())
print("Estações:", df_municipal["station_id"].nunique())

print(
    df_municipal[
        ["date", "station_id", "municipio", "codigo_ibge", "inundacao"]
    ].head(10)
)


print("Total de positivos:", df_municipal["inundacao"].sum())
print("Total de negativos:", (df_municipal["inundacao"] == 0).sum())

print("\nPositivos por município:")
print(
    df_municipal.loc[df_municipal["inundacao"] == 1]
    .groupby(["codigo_ibge", "municipio"])
    .agg(
        eventos=("inundacao", "sum"),
        primeira_data=("date", "min"),
        ultima_data=("date", "max")
    )
    .sort_values("eventos", ascending=False)
)

print("\nDatas dos eventos:")
print(
    df_municipal.loc[df_municipal["inundacao"] == 1,
                     ["date", "station_id", "municipio", "codigo_ibge"]]
    .sort_values("date")
    .to_string(index=False)
)

