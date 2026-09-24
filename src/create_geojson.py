from pathlib import Path
import geopandas as gpd


BASE_DIR = Path(__file__).resolve().parent.parent

SHP_PATH = (
    BASE_DIR
    / "data"
    / "geo"
    / "RS_Municipios_2025.shp"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "geo"
    / "rs_municipios.geojson"
)


def main():

    print("Lendo malha municipal do IBGE...")

    gdf = gpd.read_file(SHP_PATH)
    gdf = gpd.read_file(SHP_PATH)

    gdf = gdf[gdf["SIGLA_UF"] == "RS"].copy()

    print(f"Municípios RS encontrados: {len(gdf)}")

    print(f"Municípios encontrados: {len(gdf)}")
    print("\nColunas:")
    print(gdf.columns.tolist())

    print("\nCRS:")
    print(gdf.crs)

    # Mantém apenas informações úteis para o mapa
    colunas = [
        "CD_MUN",
        "NM_MUN",
        "SIGLA_UF",
        "geometry"
    ]

    gdf = gdf[colunas].copy()

    # Garante coordenadas geográficas
    gdf = gdf.to_crs("EPSG:4326")

    print("\nSalvando GeoJSON...")
    print("\nUFs:")
    print(gdf["SIGLA_UF"].value_counts())

    print("\nPrimeiros municípios:")
    print(
        gdf[
            ["CD_MUN", "NM_MUN", "SIGLA_UF"]
        ].head(10)
    )

    print("\nRegistros que não são RS:")
    print(
        gdf[gdf["SIGLA_UF"] != "RS"][
            ["CD_MUN", "NM_MUN", "SIGLA_UF"]
        ]
    )
    gdf.to_file(
        OUTPUT_PATH,
        driver="GeoJSON"
    )
    print(gdf[["CD_MUN", "NM_MUN"]].head())

    print(f"\nGeoJSON criado com sucesso:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()