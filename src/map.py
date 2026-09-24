import os
import json

import joblib
import folium
import pandas as pd
from branca.element import Element

# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GOLD_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "gold",
        "features_inundacao.parquet"
    )
)

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "model.joblib"
    )
)

GEOJSON_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "geo",
        "municipios_rs.geojson"
    )
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "mapa_inundacao_rs.html"
    )
)


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():

    df = pd.read_parquet(
        GOLD_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


def carregar_modelo():

    return joblib.load(
        MODEL_PATH
    )


def carregar_geojson():

    with open(
        GEOJSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        geojson = json.load(f)

    return geojson


# ============================================================
# NORMALIZAÇÃO DO GEOJSON
# ============================================================

def normalizar_geojson(geojson):

    """
    Garante que o código do município no GeoJSON
    esteja disponível como 'codigo_ibge'.

    O GeoJSON original utiliza CD_MUN.
    """

    for feature in geojson.get(
        "features",
        []
    ):

        properties = feature.get(
            "properties",
            {}
        )

        codigo = properties.get(
            "CD_MUN"
        )

        if codigo is None:
            continue

        try:

            codigo = str(
                int(
                    float(codigo)
                )
            )

        except (
            ValueError,
            TypeError
        ):

            codigo = str(
                codigo
            ).strip()

        properties[
            "codigo_ibge"
        ] = codigo

    return geojson


# ============================================================
# PROBABILIDADES
# ============================================================

def calcular_probabilidades(
    df,
    artifact
):

    features = artifact[
        "features"
    ]

    imputer = artifact[
        "imputer"
    ]

    model = artifact[
        "model"
    ]

    X = df[
        features
    ].copy()

    X_imputed = imputer.transform(
        X
    )

    df = df.copy()

    df[
        "probabilidade_inundacao"
    ] = (
        model.predict_proba(
            X_imputed
        )[:, 1]
    )

    return df


# ============================================================
# SELEÇÃO DA DATA
# ============================================================

def preparar_dados_data(
    df,
    data=None
):

    datas_validas = df[
        "date"
    ].dropna()

    if datas_validas.empty:

        raise ValueError(
            "A base Gold não possui datas válidas."
        )

    data_min = datas_validas.min()
    data_max = datas_validas.max()

    # --------------------------------------------------------
    # Data padrão
    # --------------------------------------------------------

    if data is None:

        data = data_max

    else:

        data = pd.to_datetime(
            data,
            errors="coerce"
        )

        if pd.isna(data):

            raise ValueError(
                "Data inválida. "
                "Use o formato AAAA-MM-DD."
            )

    # --------------------------------------------------------
    # Verificar intervalo
    # --------------------------------------------------------

    if (
        data.normalize()
        < data_min.normalize()
        or
        data.normalize()
        > data_max.normalize()
    ):

        raise ValueError(
            "Data fora do período disponível. "
            f"Escolha uma data entre "
            f"{data_min.strftime('%Y-%m-%d')} "
            f"e "
            f"{data_max.strftime('%Y-%m-%d')}."
        )

    # --------------------------------------------------------
    # Filtrar
    # --------------------------------------------------------

    df_dia = df[
        df["date"].dt.normalize()
        == data.normalize()
    ].copy()

    if df_dia.empty:

        raise ValueError(
            f"Não existem dados para "
            f"{data.strftime('%Y-%m-%d')}."
        )

    return (
        df_dia,
        data,
        data_min,
        data_max
    )


# ============================================================
# CONTROLE DE DATA NO MAPA
# ============================================================

def adicionar_controle_data(
    mapa,
    data_selecionada,
    data_min,
    data_max
):
    """
    Adiciona um controle de seleção de data
    diretamente sobre o mapa.
    """

    data_atual = data_selecionada.strftime("%Y-%m-%d")
    data_min_str = data_min.strftime("%Y-%m-%d")
    data_max_str = data_max.strftime("%Y-%m-%d")

    html = f"""
    <div>
        id="controle-data"
        style="
            position: fixed;
            top: 20px;
            left: 70px;
            z-index: 999999;
            background: white;
            padding: 12px 15px;
            border: 1px solid #999;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.30);
            font-family: Arial, sans-serif;
            min-width: 250px;
            pointer-events: auto;
        "
    >

        <div>
            style="
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 14px;
            "
        >
            Data da análise
        </div>

        <form
            method="GET"
            action="/mapa"
            style="margin: 0;"
        >

            <input
                type="date"
                name="data"
                value="{data_atual}"
                min="{data_min_str}"
                max="{data_max_str}"
                required
                style="
                    padding: 6px;
                    border: 1px solid #aaa;
                    border-radius: 4px;
                    font-size: 13px;
                "
            >

            <button
                type="submit"
                style="
                    margin-left: 5px;
                    padding: 7px 12px;
                    border: none;
                    border-radius: 4px;
                    background: #333;
                    color: white;
                    font-size: 13px;
                    cursor: pointer;
                "
            >
                Atualizar
            </button>

        </form>

        <div>
            style="
                margin-top: 7px;
                font-size: 11px;
                color: #666;
            "
        >
            Período disponível:<br>
            {data_min_str} até {data_max_str}
        </div>

    </div>
    """

    mapa.get_root().html.add_child(
        Element(html)
    )


# ============================================================
# CRIAÇÃO DO MAPA
# ============================================================

def criar_mapa(
    data=None
):

    # ========================================================
    # CARREGAR DADOS
    # ========================================================

    print(
        "Carregando dados..."
    )

    df = carregar_dados()

    print(
        f"Linhas no Gold: {len(df)}"
    )

    # ========================================================
    # CARREGAR MODELO
    # ========================================================

    print(
        "Carregando modelo..."
    )

    artifact = carregar_modelo()

    # ========================================================
    # CALCULAR PROBABILIDADES
    # ========================================================

    print(
        "Calculando probabilidades..."
    )

    df = calcular_probabilidades(
        df,
        artifact
    )

    # ========================================================
    # SELECIONAR DATA
    # ========================================================

    (
        df_dia,
        data_selecionada,
        data_min,
        data_max
    ) = preparar_dados_data(
        df,
        data
    )

    print(
        "Gerando mapa para "
        f"{data_selecionada.strftime('%d/%m/%Y')}"
    )

    print(
        f"Linhas na data selecionada: "
        f"{len(df_dia)}"
    )

    # ========================================================
    # MAPA BASE
    # ========================================================

    mapa = folium.Map(

        location=[
            -30.5,
            -53.0
        ],

        zoom_start=7,

        tiles="CartoDB positron"
    )

    # ========================================================
    # DADOS POR MUNICÍPIO
    # ========================================================

    df_geo = (

        df_dia

        .dropna(
            subset=[
                "codigo_ibge"
            ]
        )

        .groupby(
            "codigo_ibge",
            as_index=False
        )

        .agg(
            probabilidade_inundacao=(
                "probabilidade_inundacao",
                "max"
            )
        )
    )

    # ========================================================
    # NORMALIZAR CÓDIGO IBGE
    # ========================================================

    df_geo[
        "codigo_ibge"
    ] = (

        pd.to_numeric(
            df_geo[
                "codigo_ibge"
            ],
            errors="coerce"
        )

        .astype("Int64")

        .astype(str)
    )

    df_geo = df_geo[
        df_geo[
            "codigo_ibge"
        ] != "<NA>"
    ].copy()

    print(
        f"Municípios com dados: "
        f"{len(df_geo)}"
    )

    # ========================================================
    # GEOJSON
    # ========================================================

    print(
        "Carregando GeoJSON..."
    )

    geojson = carregar_geojson()

    geojson = normalizar_geojson(
        geojson
    )

    # ========================================================
    # CHOROPLETH
    # ========================================================

    folium.Choropleth(

        geo_data=geojson,

        data=df_geo,

        columns=[
            "codigo_ibge",
            "probabilidade_inundacao"
        ],

        key_on=(
            "feature.properties.codigo_ibge"
        ),

        fill_color="YlOrRd",

        fill_opacity=0.7,

        line_opacity=0.3,

        legend_name=(
            "Probabilidade estimada "
            "de inundação"
        ),

        nan_fill_color="lightgray",

        nan_fill_opacity=0.35,

        name="Probabilidade"

    ).add_to(mapa)

    # ========================================================
    # TOOLTIP DOS MUNICÍPIOS
    # ========================================================

    folium.GeoJson(

        geojson,

        name="Municípios",

        style_function=lambda feature: {

            "fillOpacity": 0,

            "color": "transparent",

            "weight": 0

        },

        highlight_function=lambda feature: {

            "weight": 2,

            "color": "#333",

            "fillOpacity": 0.05

        },

        tooltip=folium.GeoJsonTooltip(

            fields=[
                "NM_MUN",
                "CD_MUN"
            ],

            aliases=[
                "Município:",
                "Código IBGE:"
            ],

            localize=True,

            sticky=True,

            labels=True,

            style="""
                background-color: white;
                color: #333333;
                font-family: Arial;
                font-size: 12px;
                padding: 8px;
            """

        )

    ).add_to(mapa)

    # ========================================================
    # POPUPS DAS ESTAÇÕES
    # ========================================================

    camada_estacoes = (
        folium.FeatureGroup(
            name="Estações meteorológicas"
        )
    )

    for _, row in df_dia.iterrows():

        if (

            pd.isna(
                row.get("latitude")
            )

            or

            pd.isna(
                row.get("longitude")
            )

        ):

            continue

        probabilidade = (

            row[
                "probabilidade_inundacao"
            ]

            * 100

        )

        popup = criar_popup(
            row
        )

        folium.CircleMarker(

            location=[

                row["latitude"],

                row["longitude"]

            ],

            radius=6,

            popup=folium.Popup(

                popup,

                max_width=320

            ),

            tooltip=(

                f"{row['municipio']} | "
                f"{probabilidade:.1f}%"

            ),

            color="black",

            weight=1,

            fill=True,

            fill_opacity=0.9

        ).add_to(
            camada_estacoes
        )

    camada_estacoes.add_to(
        mapa
    )

    # ========================================================
    # CONTROLE DE DATA
    # ========================================================

    adicionar_controle_data(

        mapa,

        data_selecionada,

        data_min,

        data_max

    )

    # ========================================================
    # CONTROLE DE CAMADAS
    # ========================================================

    folium.LayerControl().add_to(
        mapa
    )

    # ========================================================
    # SALVAR
    # ========================================================

    mapa.save(
        OUTPUT_PATH
    )

    print()

    print(
        "======================================"
    )

    print(
        "MAPA GERADO COM SUCESSO"
    )

    print(
        "======================================"
    )

    print(
        f"Data: "
        f"{data_selecionada.strftime('%d/%m/%Y')}"
    )

    print(
        f"Municípios com dados: "
        f"{len(df_geo)}"
    )

    print(
        f"Arquivo: "
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # RETORNAR MAPA
    # ========================================================

    return mapa

# ============================================================
# POPUP
# ============================================================

def criar_popup(row):
    probabilidade = (
        row["probabilidade_inundacao"] * 100
    )

    def valor(valor, casas=1, sufixo=""):
        if pd.isna(valor):
            return "N/D"

        return (
            f"{valor:.{casas}f}"
            f"{sufixo}"
        )

    municipio = row.get(
        "municipio",
        "Município não identificado"
    )

    codigo_ibge = row.get(
        "codigo_ibge",
        "N/D"
    )

    return f"""
    <div style="width: 280px">

        <h4>{municipio}</h4>

        <b>Probabilidade estimada:</b><br>
        {probabilidade:.1f}%

        <br><br>

        <b>Código IBGE:</b>
        {codigo_ibge}

        <br><br>

        <b>Chuva acumulada</b><br>

        24h:
        {valor(row.get("precip_sum_24h"))} mm<br>

        72h:
        {valor(row.get("precip_sum_72h"))} mm<br>

        7 dias:
        {valor(row.get("precip_sum_7d"))} mm

        <br><br>

        <b>Condições meteorológicas</b><br>

        Temperatura média:
        {valor(row.get("temperature_avg"))} °C<br>

        Umidade média:
        {valor(row.get("humidity_avg"))} %<br>

        Pressão:
        {valor(row.get("pressure_avg"))} hPa<br>

        Variação da pressão em 24h:
        {valor(row.get("pressure_change_24h"))} hPa

        <br><br>

        <b>Data:</b>
        {pd.to_datetime(row["date"]).strftime("%d/%m/%Y")}

    </div>
    """
# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    criar_mapa()