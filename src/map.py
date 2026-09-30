import os
import json

import joblib
import folium
import pandas as pd
import copy

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
        "model_calibrado.joblib"
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

FORECAST_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "forecast",
        "probabilidades_inundacao_16d.parquet"
    )
)

RANKING_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "forecast",
        "ranking_municipios_16d.parquet"
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

    modelo = joblib.load(
        MODEL_PATH
    )

    if not isinstance(modelo, dict):
        raise TypeError(
            "O model.joblib deveria conter um dicionário."
        )

    if "features" not in modelo:
        raise KeyError(
            "A chave 'features' não existe no modelo."
        )

    if "model" not in modelo:
        raise KeyError(
            "A chave 'model' não existe no modelo."
        )

    return modelo


def carregar_geojson():

    with open(
        GEOJSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        geojson = json.load(f)

    return geojson


def carregar_previsao():

    if not os.path.exists(
        FORECAST_PATH
    ):
        raise FileNotFoundError(
            f"Arquivo de previsão não encontrado:\n"
            f"{FORECAST_PATH}"
        )

    df = pd.read_parquet(
        FORECAST_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["codigo_ibge"] = (
        pd.to_numeric(
            df["codigo_ibge"],
            errors="coerce"
        )
        .astype("Int64")
    )

    return df


def carregar_ranking():

    if not os.path.exists(
        RANKING_PATH
    ):
        raise FileNotFoundError(
            f"Ranking não encontrado:\n"
            f"{RANKING_PATH}"
        )

    ranking = pd.read_parquet(
        RANKING_PATH
    )

    ranking["codigo_ibge"] = (
        pd.to_numeric(
            ranking["codigo_ibge"],
            errors="coerce"
        )
        .astype("Int64")
    )

    ranking["data_maior_probabilidade"] = (
        pd.to_datetime(
            ranking[
                "data_maior_probabilidade"
            ],
            errors="coerce"
        )
    )

    return ranking


# ============================================================
# NORMALIZAÇÃO DO GEOJSON
# ============================================================

def normalizar_geojson(
    geojson
):

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
# PROBABILIDADES HISTÓRICAS
# ============================================================

def calcular_probabilidades(
    df,
    artifact
):

    features = artifact[
        "features"
    ]

    model = artifact[
        "model"
    ]

    X = df[
        features
    ].copy()

    df = df.copy()

    df[
        "probabilidade_inundacao"
    ] = (
        model.predict_proba(X)[:, 1]
    )

    return df


# ============================================================
# SELEÇÃO DA DATA HISTÓRICA
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
# POPUP HISTÓRICO
# ============================================================

def criar_popup(
    row
):

    probabilidade = (
        row[
            "probabilidade_inundacao"
        ] * 100
    )

    def valor(
        valor,
        casas=1,
        sufixo=""
    ):

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

        <b>Probabilidade estimada para D+1:</b><br>
        {probabilidade:.3f}%

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
        {pd.to_datetime(
            row["date"]
        ).strftime("%d/%m/%Y")}

    </div>
    """


# ============================================================
# POPUP PREVISÃO
# ============================================================

def criar_popup_previsao(
    row,
    dia_previsao
):

    probabilidade = (
        row[
            "probabilidade_inundacao"
        ] * 100
    )

    municipio = row.get(
        "municipio",
        "Município não identificado"
    )

    codigo_ibge = row.get(
        "codigo_ibge",
        "N/D"
    )

    station_id = row.get(
        "station_id",
        "N/D"
    )

    data = pd.to_datetime(
        row["date"]
    )

    return f"""
    <div style="width: 280px">

        <h4>{municipio}</h4>

        <b>Previsão de inundação</b><br>

        <b>D+{dia_previsao}</b> —
        {data.strftime("%d/%m/%Y")}

        <br><br>

        <b>Probabilidade estimada:</b><br>

        <span style="
            font-size: 20px;
            font-weight: bold;
        ">
            {probabilidade:.4f}%
        </span>

        <br><br>

        <b>Código IBGE:</b>
        {codigo_ibge}

        <br><br>

        <b>Estação:</b>
        {station_id}

        <br><br>

        <span style="
            font-size: 11px;
            color: #666;
        ">
            Estimativa baseada na previsão meteorológica
            e no modelo calibrado de inundação.
        </span>

    </div>
    """


# ============================================================
# TOP 5
# ============================================================

def adicionar_top5(
    mapa,
    ranking
):

    top5 = (
        ranking
        .sort_values(
            "probabilidade_acumulada_16d",
            ascending=False
        )
        .head(5)
        .copy()
    )

    linhas = ""

    for _, row in top5.iterrows():

        municipio = row[
            "municipio"
        ]

        prob = row[
            "probabilidade_acumulada_16d_pct"
        ]

        linhas += f"""
        <div style="
            margin-bottom: 6px;
            padding-bottom: 6px;
            border-bottom: 1px solid #ddd;
        ">

            <b>{int(row["posicao"])}.
            {municipio}</b><br>

            <span style="font-size: 13px;">
                Acumulado:
                <b>{prob:.4f}%</b>
            </span>

        </div>
        """

    html = f"""
    <div id="painel-top5"
        style="
            position: fixed;
            top: 110px;
            right: 20px;
            z-index: 999999;

            background: white;

            padding: 12px 15px;

            border: 1px solid #999;
            border-radius: 8px;

            box-shadow:
                0 2px 8px
                rgba(0,0,0,0.30);

            font-family: Arial, sans-serif;

            width: 220px;

            display: none;
        "
    >

        <div style="
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 4px;
        ">
            Top 5 — 16 dias
        </div>

        <div style="
            font-size: 10px;
            color: #666;
            margin-bottom: 10px;
        ">
            Probabilidade acumulada estimada
        </div>

        {linhas}

        <div style="
            font-size: 10px;
            color: #666;
            margin-top: 8px;
        ">
            * Cálculo considerando independência
            entre os dias.
        </div>

    </div>
    """

    mapa.get_root().html.add_child(
        Element(html)
    )


# ============================================================
# CONTROLE PRINCIPAL
# ============================================================

def adicionar_controle_modo(
    mapa
):

    mapa_name = mapa.get_name()

    html = f"""
    <div
        id="controle-modo"
        style="
            position: fixed;
            top: 20px;
            left: 70px;
            z-index: 999999;

            background: white;

            padding: 12px 15px;

            border: 1px solid #999;
            border-radius: 8px;

            box-shadow:
                0 2px 8px
                rgba(0,0,0,0.30);

            font-family: Arial, sans-serif;

            min-width: 260px;
        "
    >

        <div style="
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 14px;
        ">
            Modo da análise
        </div>

        <select
            id="modo-mapa"
            style="
                width: 100%;
                padding: 6px;
                border: 1px solid #aaa;
                border-radius: 4px;
                font-size: 13px;
            "
        >

            <option value="historico">
                Histórico
            </option>

            <option value="previsao">
                Previsão — 16 dias
            </option>

        </select>

        <div
            id="controle-previsao"
            style="
                display: none;
                margin-top: 10px;
            "
        >

            <label
                for="data-previsao"
                style="
                    font-size: 12px;
                    font-weight: bold;
                "
            >
                Dia da previsão
            </label>

            <select
                id="data-previsao"
                style="
                    width: 100%;
                    margin-top: 5px;
                    padding: 6px;
                    border: 1px solid #aaa;
                    border-radius: 4px;
                    font-size: 13px;
                "
            >
    """

    html += """
            </select>

        </div>

    </div>
    """

    mapa.get_root().html.add_child(
        Element(html)
    )


# ============================================================
# CONTROLE DA PREVISÃO
# ============================================================

def adicionar_javascript_previsao(
    mapa,
    camadas_previsao,
    camada_acumulada,
    dados_datas
):

    mapa_name = mapa.get_name()

    nomes_camadas = [
        camada.get_name()
        for camada in camadas_previsao
    ]

    acumulada_name = (
        camada_acumulada.get_name()
    )

    datas_json = json.dumps(
        dados_datas,
        ensure_ascii=False
    )

    camadas_json = json.dumps(
        nomes_camadas
    )

    js = f"""
    <script>

    document.addEventListener(
        "DOMContentLoaded",
        function() {{

            const mapa =
                {mapa_name};

            const modo =
                document.getElementById(
                    "modo-mapa"
                );

            const controlePrevisao =
                document.getElementById(
                    "controle-previsao"
                );

            const dataPrevisao =
                document.getElementById(
                    "data-previsao"
                );

            const painelTop5 =
                document.getElementById(
                    "painel-top5"
                );

            const camadas =
                {camadas_json};

            const camadaAcumulada =
                {acumulada_name};

            const datas =
                {datas_json};

            // ----------------------------------------
            // Preencher seletor D+1 ... D+16
            // ----------------------------------------

            datas.forEach(
                function(item) {{

                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        item.layer_index;

                    option.textContent =
                        "D+" +
                        item.dia +
                        " — " +
                        item.data;

                    dataPrevisao.appendChild(
                        option
                    );
                }}
            );

            function esconderPrevisoes() {{

                camadas.forEach(
                    function(layerName) {{

                        const layer =
                            window[layerName];

                        if (
                            layer &&
                            mapa.hasLayer(layer)
                        ) {{
                            mapa.removeLayer(
                                layer
                            );
                        }}
                    }}
                );

                const acumulada =
                    window[camadaAcumulada];

                if (
                    acumulada &&
                    mapa.hasLayer(acumulada)
                ) {{
                    mapa.removeLayer(
                        acumulada
                    );
                }}
            }}

            function mostrarPrevisao(
                indice
            ) {{

                esconderPrevisoes();

                const layer =
                    window[
                        camadas[indice]
                    ];

                if (layer) {{
                    mapa.addLayer(
                        layer
                    );
                }}
            }}

            function mudarModo() {{

                if (
                    modo.value ===
                    "previsao"
                ) {{

                    controlePrevisao.style.display =
                        "block";

                    painelTop5.style.display =
                        "block";

                    mostrarPrevisao(
                        Number(
                            dataPrevisao.value
                        )
                    );

                }} else {{

                    controlePrevisao.style.display =
                        "none";

                    painelTop5.style.display =
                        "none";

                    esconderPrevisoes();
                }}
            }}

            modo.addEventListener(
                "change",
                mudarModo
            );

            dataPrevisao.addEventListener(
                "change",
                function() {{

                    if (
                        modo.value ===
                        "previsao"
                    ) {{

                        mostrarPrevisao(
                            Number(
                                dataPrevisao.value
                            )
                        );
                    }}
                }}
            );

            // ----------------------------------------
            // Estado inicial
            // ----------------------------------------

            dataPrevisao.value = "0";

            esconderPrevisoes();

        }}
    );

    </script>
    """

    mapa.get_root().html.add_child(
        Element(js)
    )


# ============================================================
# CRIAR CAMADA DE PREVISÃO
# ============================================================

def criar_camada_previsao(
    mapa,
    geojson,
    df_dia,
    dia_previsao,
    data_previsao
):
    geojson = copy.deepcopy(geojson)

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

    # --------------------------------------------------------
    # Dicionário código IBGE → probabilidade
    # --------------------------------------------------------

    probabilidades = dict(
        zip(
            df_geo[
                "codigo_ibge"
            ],
            df_geo[
                "probabilidade_inundacao"
            ]
        )
    )

    for feature in geojson["features"]:
        codigo = str(
            feature["properties"].get(
                "codigo_ibge",
                ""
            )
        )

        probabilidade = probabilidades.get(codigo)

        if probabilidade is None:
            feature["properties"]["probabilidade_pct"] = "N/D"
        else:
            feature["properties"]["probabilidade_pct"] = (
                f"{probabilidade * 100:.4f}%"
            )
    # --------------------------------------------------------
    # Camada
    # --------------------------------------------------------

    grupo = folium.FeatureGroup(
        name=(
            f"D+{dia_previsao} — "
            f"{data_previsao.strftime('%d/%m/%Y')}"
        ),
        show=False
    )

    # --------------------------------------------------------
    # Função de cor
    # --------------------------------------------------------

    def obter_cor(
        probabilidade
    ):

        if probabilidade is None:
            return "#d3d3d3"

        pct = probabilidade * 100

        if pct < 0.1:
            return "#ffffcc"

        if pct < 0.25:
            return "#ffeda0"

        if pct < 0.5:
            return "#fed976"

        if pct < 1.0:
            return "#feb24c"

        if pct < 2.0:
            return "#fd8d3c"

        if pct < 5.0:
            return "#f03b20"

        return "#bd0026"

    # --------------------------------------------------------
    # GeoJSON
    # --------------------------------------------------------

    def style_function(
        feature
    ):

        codigo = str(
            feature[
                "properties"
            ].get(
                "codigo_ibge",
                ""
            )
        )

        probabilidade = (
            probabilidades.get(
                codigo
            )
        )

        return {
            "fillColor":
                obter_cor(
                    probabilidade
                ),
            "color":
                "#666666",
            "weight":
                0.5,
            "fillOpacity":
                0.7
        }

    # --------------------------------------------------------
    # Tooltip
    # --------------------------------------------------------

    tooltip = folium.GeoJsonTooltip(
    fields=[
        "NM_MUN",
        "CD_MUN",
        "probabilidade_pct"
    ],
    aliases=[
        "Município:",
        "Código IBGE:",
        "Probabilidade:"
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

    folium.GeoJson(
        geojson,
        style_function=style_function,
        highlight_function=lambda feature: {
            "weight": 2,
            "color": "#333",
            "fillOpacity": 0.8
        },
        tooltip=tooltip
    ).add_to(
        grupo
    )

    grupo.add_to(
        mapa
    )

    return grupo


# ============================================================
# CAMADA ACUMULADA
# ============================================================

def criar_camada_acumulada(
    mapa,
    geojson,
    ranking
):

    probabilidades = {}

    for _, row in ranking.iterrows():

        codigo = str(
            int(
                row[
                    "codigo_ibge"
                ]
            )
        )

        probabilidades[
            codigo
        ] = row[
            "probabilidade_acumulada_16d"
        ]

    grupo = folium.FeatureGroup(
        name="Acumulado — 16 dias",
        show=False
    )

    def obter_cor(
        probabilidade
    ):

        if probabilidade is None:
            return "#d3d3d3"

        pct = probabilidade * 100

        if pct < 1:
            return "#ffffcc"

        if pct < 2:
            return "#ffeda0"

        if pct < 3:
            return "#fed976"

        if pct < 5:
            return "#feb24c"

        if pct < 7:
            return "#fd8d3c"

        return "#e31a1c"

    def style_function(
        feature
    ):

        codigo = str(
            feature[
                "properties"
            ].get(
                "codigo_ibge",
                ""
            )
        )

        probabilidade = (
            probabilidades.get(
                codigo
            )
        )

        return {
            "fillColor":
                obter_cor(
                    probabilidade
                ),
            "color":
                "#666666",
            "weight":
                0.5,
            "fillOpacity":
                0.7
        }

    folium.GeoJson(
        geojson,
        style_function=style_function,
        highlight_function=lambda feature: {
            "weight": 2,
            "color": "#333",
            "fillOpacity": 0.8
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
            localize=False,
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
    ).add_to(
        grupo
    )

    grupo.add_to(
        mapa
    )

    return grupo


# ============================================================
# MAPA
# ============================================================

def criar_mapa(
    data=None
):

    # ========================================================
    # HISTÓRICO
    # ========================================================

    print(
        "Carregando dados históricos..."
    )

    df = carregar_dados()

    print(
        f"Linhas no Gold: {len(df)}"
    )

    print(
        "Carregando modelo..."
    )

    artifact = carregar_modelo()

    print(
        "Calculando probabilidades históricas..."
    )

    df = calcular_probabilidades(
        df,
        artifact
    )

    (
        df_dia,
        data_selecionada,
        data_min,
        data_max
    ) = preparar_dados_data(
        df,
        data
    )

    # ========================================================
    # PREVISÃO
    # ========================================================

    print(
        "Carregando previsão de 16 dias..."
    )

    df_forecast = carregar_previsao()

    print(
        f"Registros de previsão: "
        f"{len(df_forecast):,}"
    )

    print(
        "Carregando ranking..."
    )

    ranking = carregar_ranking()

    print(
        f"Municípios no ranking: "
        f"{len(ranking)}"
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
    # MAPA BASE
    # ========================================================

    mapa = folium.Map(
        location=[
            -30.5,
            -53.0
        ],
        zoom_start=7,
        tiles=None
    )

    # ========================================================
    # TÍTULO
    # ========================================================

    titulo = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);

        z-index: 9999;

        background-color: white;

        padding: 10px 18px;

        border: 1px solid #999;
        border-radius: 6px;

        box-shadow:
            0 2px 6px
            rgba(0,0,0,0.25);

        font-family: Arial, sans-serif;

        font-size: 18px;
        font-weight: bold;
    ">

        Probabilidade estimada de inundação
        — Rio Grande do Sul

    </div>
    """

    mapa.get_root().html.add_child(
        Element(titulo)
    )

    # ========================================================
    # SUBTÍTULO
    # ========================================================

    subtitulo = f"""
    <div
        id="subtitulo-mapa"
        style="
            position: fixed;

            top: 58px;
            left: 50%;

            transform:
                translateX(-50%);

            z-index: 9998;

            background-color: white;

            padding: 6px 12px;

            border: 1px solid #bbb;
            border-radius: 5px;

            font-family: Arial, sans-serif;

            font-size: 12px;
        "
    >

        Histórico:
        estimativa para o dia seguinte
        (D+1), baseada nas condições
        observadas na data selecionada.

    </div>
    """

    mapa.get_root().html.add_child(
        Element(subtitulo)
    )

    # ========================================================
    # OBSERVAÇÃO
    # ========================================================

    info = """
    <div
        id="info-mapa"
        style="
            position: fixed;

            bottom: 20px;
            left: 20px;

            z-index: 9999;

            background-color: white;

            padding: 10px 14px;

            border: 1px solid #999;
            border-radius: 5px;

            font-family: Arial, sans-serif;

            font-size: 12px;

            box-shadow:
                0 2px 5px
                rgba(0,0,0,0.20);
        "
    >

        <b>Observação</b><br>

        Municípios em cinza não possuem
        estimativa disponível para a
        data selecionada.

    </div>
    """

    mapa.get_root().html.add_child(
        Element(info)
    )

    # ========================================================
    # HISTÓRICO — DADOS POR MUNICÍPIO
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

    # ========================================================
    # HISTÓRICO — CHOROPLETH
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
            "de inundação no dia seguinte"
        ),
        nan_fill_color="lightgray",
        nan_fill_opacity=0.35,
        name="Histórico — Probabilidade D+1"
    ).add_to(
        mapa
    )

    # ========================================================
    # TOOLTIP HISTÓRICO
    # ========================================================

    folium.GeoJson(
        geojson,
        name="Municípios — Histórico",
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
    ).add_to(
        mapa
    )

    # ========================================================
    # ESTAÇÕES HISTÓRICAS
    # ========================================================

    camada_estacoes = folium.FeatureGroup(
        name="Estações meteorológicas — Histórico"
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
            ] * 100
        )

        popup = criar_popup(
            row
        )

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=3,
            popup=folium.Popup(
                popup,
                max_width=320
            ),
            tooltip=(
                f"{row['municipio']} | "
                f"{probabilidade:.3f}%"
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
    # CAMADAS DE PREVISÃO
    # ========================================================

    df_forecast = (
        df_forecast
        .dropna(
            subset=[
                "codigo_ibge"
            ]
        )
        .copy()
    )

    datas_forecast = (
        df_forecast[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    camadas_previsao = []
    dados_datas = []

    print()
    print(
        "Criando camadas de previsão..."
    )

    for indice, data_prev in enumerate(
        datas_forecast,
        start=1
    ):

        df_dia_forecast = (
            df_forecast[
                df_forecast[
                    "date"
                ].dt.normalize()
                ==
                data_prev.normalize()
            ].copy()
        )

        camada = criar_camada_previsao(
            mapa=mapa,
            geojson=geojson,
            df_dia=df_dia_forecast,
            dia_previsao=indice,
            data_previsao=data_prev
        )

        camadas_previsao.append(
            camada
        )

        dados_datas.append(
            {
                "dia": indice,
                "data": data_prev.strftime(
                    "%d/%m/%Y"
                ),
                "layer_index": indice - 1
            }
        )

    # ========================================================
    # CAMADA ACUMULADA
    # ========================================================

    camada_acumulada = (
        criar_camada_acumulada(
            mapa=mapa,
            geojson=geojson,
            ranking=ranking
        )
    )

    # ========================================================
    # ESTAÇÕES DA PREVISÃO
    # ========================================================

    camada_estacoes_forecast = (
        folium.FeatureGroup(
            name="Estações meteorológicas — Previsão",
            show=False
        )
    )

    for _, row in df_forecast.iterrows():

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

        data_prev = pd.to_datetime(
            row["date"]
        )

        dia_previsao = (
            datas_forecast[
                datas_forecast
                <= data_prev
            ].shape[0]
        )

        popup = criar_popup_previsao(
            row,
            dia_previsao
        )

        probabilidade = (
            row[
                "probabilidade_inundacao"
            ] * 100
        )

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=3,
            popup=folium.Popup(
                popup,
                max_width=320
            ),
            tooltip=(
                f"{row['municipio']} | "
                f"D+{dia_previsao} | "
                f"{probabilidade:.4f}%"
            ),
            color="black",
            weight=1,
            fill=True,
            fill_opacity=0.9
        ).add_to(
            camada_estacoes_forecast
        )

    camada_estacoes_forecast.add_to(
        mapa
    )
    

    # ========================================================
    # TOP 5
    # ========================================================

    adicionar_top5(
        mapa,
        ranking
    )

    # ========================================================
    # CONTROLE DE MODO
    # ========================================================

    adicionar_controle_modo(
        mapa
    )

    adicionar_javascript_previsao(
        mapa,
        camadas_previsao,
        camada_acumulada,
        dados_datas
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
        f"Histórico: "
        f"{data_selecionada.strftime('%d/%m/%Y')}"
    )

    print(
        f"Previsão: "
        f"{len(datas_forecast)} dias"
    )

    print(
        f"Municípios históricos: "
        f"{len(df_geo)}"
    )

    print(
        f"Municípios no ranking: "
        f"{len(ranking)}"
    )

    print(
        f"Arquivo: "
        f"{OUTPUT_PATH}"
    )

    print(
        "======================================"
    )

    return mapa


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    criar_mapa()