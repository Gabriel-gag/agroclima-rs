"""
Transformação - AgroClima RS

Bronze -> Silver -> Gold

Escopo:
    Todas as estações meteorológicas automáticas
    do Rio Grande do Sul encontradas no Bronze.

Target:
    inundacao

O target é definido a partir do Atlas de Desastres,
considerando COBRADE 12100.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# UTILIDADES
# ============================================================

def _normalize_text(value):
    """
    Normaliza texto para facilitar joins entre:
    - INMET
    - Atlas de Desastres
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    # remove acentos
    replacements = {
        "Á": "A",
        "À": "A",
        "Ã": "A",
        "Â": "A",
        "Ä": "A",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "Ó": "O",
        "Ò": "O",
        "Õ": "O",
        "Ô": "O",
        "Ö": "O",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "Ç": "C",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def _normalize_station_municipality(value):
    """
    Converte nomes de estação em um município-base.

    Exemplos:

        PORTO ALEGRE - JARDIM BOTANICO
        -> PORTO ALEGRE

        CAXIAS DO SUL - CRIUVA
        -> CAXIAS DO SUL

        PELOTAS / CAPAO DO LEAO
        -> PELOTAS

    A informação original permanece preservada em
    `municipio_estacao`.
    """

    value = _normalize_text(value)

    if " / " in str(value):
        value = str(value).split(" / ")[0]

    if " - " in str(value):
        value = str(value).split(" - ")[0]

    return value.strip()


def _find_column(columns, keywords):
    """
    Procura uma coluna utilizando palavras-chave.
    """

    normalized = {
        col: _normalize_text(col)
        for col in columns
    }

    for keyword in keywords:

        keyword = _normalize_text(keyword)

        for original, clean in normalized.items():

            if keyword in clean:
                return original

    return None


def _read_inmet_file(path):
    """
    Lê um CSV histórico do INMET.

    O formato dos arquivos históricos contém
    aproximadamente 8 linhas de metadados antes
    da tabela meteorológica.
    """

    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=8,
        decimal=",",
        low_memory=False
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Remove colunas completamente vazias
    df = df.dropna(
        axis=1,
        how="all"
    )

    return df


def _extract_station_id(path, df=None):
    """
    Extrai o código WMO do nome do arquivo.

    Exemplo:

    INMET_S_RS_A803_SANTA MARIA_...
    -> A803
    """

    match = re.search(
        r"_RS_([A-Z]\d{3})_",
        Path(path).name.upper()
    )

    if match:
        return match.group(1)

    # fallback: procura código nas colunas
    if df is not None:

        for col in df.columns:

            if "CODIGO" in _normalize_text(col):

                values = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                if len(values) > 0:
                    return values.iloc[0]

    return None


def _extract_station_name(path):
    """
    Extrai o nome da estação do filename.

    Exemplo:

    INMET_S_RS_A803_SANTA MARIA_01-01-2025...
    """

    name = Path(path).stem.upper()

    match = re.search(
        r"_RS_[A-Z]\d{3}_(.*?)_\d{2}-\d{2}-\d{4}",
        name
    )

    if match:
        return match.group(1).strip()

    return None


def _build_datetime(df):
    """
    Cria datetime a partir das colunas Data + Hora UTC.
    """

    date_col = _find_column(
        df.columns,
        ["DATA"]
    )

    time_col = _find_column(
        df.columns,
        ["HORA UTC", "HORA"]
    )

    if date_col is None:
        raise ValueError(
            "Coluna de data não encontrada."
        )

    if time_col is None:
        raise ValueError(
            "Coluna de hora não encontrada."
        )

    date = (
        df[date_col]
        .astype(str)
        .str.strip()
    )

    time = (
        df[time_col]
        .astype(str)
        .str.strip()
        .str.replace(
            "UTC",
            "",
            regex=False
        )
        .str.strip()
    )

    datetime = pd.to_datetime(
        date + " " + time,
        format="%Y/%m/%d %H%M",
        errors="coerce"
    )

    # fallback
    if datetime.isna().all():

        datetime = pd.to_datetime(
            date + " " + time,
            errors="coerce"
        )

    return datetime


def _numeric_series(df, keywords):
    """
    Retorna uma série numérica para a primeira
    coluna encontrada.
    """

    col = _find_column(
        df.columns,
        keywords
    )

    if col is None:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64"
        )

    values = (
        df[col]
        .astype(str)
        .str.strip()
        .replace(
            {
                "9999": np.nan,
                "9999.0": np.nan,
                "NULL": np.nan,
                "NAN": np.nan,
                "": np.nan
            }
        )
    )

    return pd.to_numeric(
        values.str.replace(
            ",",
            ".",
            regex=False
        ),
        errors="coerce"
    )


# ============================================================
# SILVER
# ============================================================

def transform_bronze_to_silver(
    bronze_dir="data/bronze",
    output_path="data/silver/fact_weather_daily.parquet"
):
    """
    Bronze -> Silver.

    Processa TODAS as estações RS encontradas no Bronze.

    Não exige que todas possuam codigo_ibge.
    """

    bronze_dir = Path(bronze_dir)
    output_path = Path(output_path)

    if not bronze_dir.exists():
        raise FileNotFoundError(
            f"Bronze não encontrada: {bronze_dir}"
        )

    files = sorted(
        [
            p
            for p in bronze_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
            and "_RS_" in p.name.upper()
        ]
    )

    if not files:
        raise FileNotFoundError(
            "Nenhum CSV de estação RS encontrado "
            "em data/bronze."
        )

    daily_frames = []

    print(
        f"\nEncontrados {len(files)} arquivos "
        f"de estações RS."
    )

    for path in files:

        try:

            df = _read_inmet_file(path)

            station_id = _extract_station_id(
                path,
                df
            )

            station_name = _extract_station_name(
                path
            )

            if station_id is None:
                print(
                    f"   ⚠ Ignorado: código não identificado "
                    f"em {path.name}"
                )
                continue

            df["datetime"] = _build_datetime(df)

            df = df[
                df["datetime"].notna()
            ].copy()

            if df.empty:
                print(
                    f"   ⚠ Sem dados válidos: "
                    f"{path.name}"
                )
                continue

            df["date"] = (
                df["datetime"]
                .dt.normalize()
            )

            # ------------------------------------------------
            # Variáveis meteorológicas
            # ------------------------------------------------

            df["precip_mm"] = _numeric_series(
                df,
                [
                    "PRECIPITAÇÃO TOTAL",
                    "PRECIPITACAO TOTAL",
                    "PRECIPITAÇÃO",
                    "PRECIPITACAO"
                ]
            )

            df["temperature_c"] = _numeric_series(
                df,
                [
                    "TEMPERATURA DO AR",
                    "TEMPERATURA"
                ]
            )

            df["humidity_pct"] = _numeric_series(
                df,
                [
                    "UMIDADE RELATIVA DO AR",
                    "UMIDADE"
                ]
            )

            df["pressure_hpa"] = _numeric_series(
                df,
                [
                    "PRESSAO ATMOSFERICA",
                    "PRESSÃO ATMOSFÉRICA",
                    "PRESSAO"
                ]
            )

            df["wind_speed_ms"] = _numeric_series(
                df,
                [
                    "VENTO, VELOCIDADE",
                    "VENTO VELOCIDADE",
                    "VELOCIDADE DO VENTO"
                ]
            )

            df["solar_radiation"] = _numeric_series(
                df,
                [
                    "RADIACAO GLOBAL",
                    "RADIAÇÃO GLOBAL"
                ]
            )

            # ------------------------------------------------
            # Agregação diária
            # ------------------------------------------------

            daily = (
                df.groupby("date")
                .agg(
                    precip_mm=(
                        "precip_mm",
                        "sum"
                    ),

                    temperature_min=(
                        "temperature_c",
                        "min"
                    ),

                    temperature_max=(
                        "temperature_c",
                        "max"
                    ),

                    temperature_avg=(
                        "temperature_c",
                        "mean"
                    ),

                    humidity_avg=(
                        "humidity_pct",
                        "mean"
                    ),

                    pressure_avg=(
                        "pressure_hpa",
                        "mean"
                    ),

                    wind_speed_avg=(
                        "wind_speed_ms",
                        "mean"
                    ),

                    solar_radiation=(
                        "solar_radiation",
                        "sum"
                    )
                )
                .reset_index()
            )

            # ------------------------------------------------
            # Metadados
            # ------------------------------------------------

            daily["station_id"] = station_id

            daily["municipio_estacao"] = (
                station_name
            )

            daily["municipio"] = (
                daily["municipio_estacao"]
                .map(
                    _normalize_station_municipality
                )
            )

            daily["codigo_ibge"] = pd.NA

            daily_frames.append(
                daily
            )

            print(
                f"   ✓ {path.name}: "
                f"{len(daily)} registros diários"
            )

        except Exception as exc:

            print(
                f"   ❌ Erro em {path.name}: "
                f"{exc}"
            )

    if not daily_frames:
        raise RuntimeError(
            "Nenhuma estação pôde ser processada."
        )

    result = pd.concat(
        daily_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "station_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_parquet(
        output_path,
        index=False
    )

    csv_path = output_path.with_suffix(
        ".csv"
    )

    result.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSilver criada:"
        f"\n  {output_path}"
        f"\n  {csv_path}"
        f"\n  Registros: {len(result)}"
        f"\n  Estações: {result['station_id'].nunique()}"
    )

    return result


# ============================================================
# ATLAS
# ============================================================

def _load_atlas(atlas_path):
    """
    Carrega o Atlas de Desastres.

    Utiliza a planilha:
        Atlas Valores Originais
    """

    atlas_path = Path(atlas_path)

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas não encontrado: {atlas_path}"
        )

    df = pd.read_excel(
        atlas_path,
        sheet_name="Atlas Valores Originais"
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def _create_municipality_ibge_lookup(
    df_atlas
):
    """
    Cria um dicionário:

        município normalizado -> código IBGE

    utilizando o próprio Atlas.

    Assim não precisamos manter manualmente
    uma lista de todos os municípios do RS.
    """

    required = [
        "Nome_Municipio",
        "Cod_IBGE_Mun"
    ]

    missing = [
        c
        for c in required
        if c not in df_atlas.columns
    ]

    if missing:
        raise ValueError(
            "Colunas ausentes no Atlas: "
            + ", ".join(missing)
        )

    lookup = (
        df_atlas[
            [
                "Nome_Municipio",
                "Cod_IBGE_Mun"
            ]
        ]
        .dropna(
            subset=[
                "Nome_Municipio"
            ]
        )
        .copy()
    )

    lookup["municipio_norm"] = (
        lookup["Nome_Municipio"]
        .map(_normalize_text)
    )

    lookup["codigo_ibge"] = pd.to_numeric(
        lookup["Cod_IBGE_Mun"],
        errors="coerce"
    )

    lookup = lookup.dropna(
        subset=[
            "municipio_norm",
            "codigo_ibge"
        ]
    )

    lookup = (
        lookup[
            [
                "municipio_norm",
                "codigo_ibge"
            ]
        ]
        .drop_duplicates(
            subset=["municipio_norm"]
        )
    )

    return dict(
        zip(
            lookup["municipio_norm"],
            lookup["codigo_ibge"]
        )
    )


def create_flood_target(
    df_silver,
    atlas_path
):
    """
    Cria o target `inundacao`.

    Target = 1 quando existe evento COBRADE 12100
    para o município na data.

    Target = 0 caso contrário.

    O município é utilizado como chave principal.
    O codigo_ibge é preenchido a partir do Atlas
    quando possível.
    """

    df = df_silver.copy()

    df_atlas = _load_atlas(
        atlas_path
    )

    # --------------------------------------------------------
    # Normalização do Atlas
    # --------------------------------------------------------

    df_atlas["municipio_norm"] = (
        df_atlas["Nome_Municipio"]
        .map(_normalize_text)
    )

    df_atlas["Data_Evento"] = pd.to_datetime(
        df_atlas["Data_Evento"],
        errors="coerce"
    ).dt.normalize()

    df_atlas["Cod_Cobrade"] = pd.to_numeric(
        df_atlas["Cod_Cobrade"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Apenas inundações
    # COBRADE 12100
    # --------------------------------------------------------

    inundacoes = df_atlas[
        df_atlas["Cod_Cobrade"] == 12100
    ].copy()

    inundacoes = inundacoes[
        [
            "municipio_norm",
            "Data_Evento",
            "Cod_IBGE_Mun"
        ]
    ].drop_duplicates()

    inundacoes["inundacao"] = 1

    inundacoes = inundacoes.rename(
        columns={
            "Data_Evento": "date"
        }
    )

    # --------------------------------------------------------
    # Município da estação
    # --------------------------------------------------------

    df["municipio_norm"] = (
        df["municipio"]
        .map(_normalize_text)
    )

    # --------------------------------------------------------
    # Código IBGE
    # --------------------------------------------------------

    ibge_lookup = (
        _create_municipality_ibge_lookup(
            df_atlas
        )
    )

    df["codigo_ibge"] = (
        df["municipio_norm"]
        .map(ibge_lookup)
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df = df.merge(
        inundacoes[
            [
                "municipio_norm",
                "date",
                "inundacao"
            ]
        ],
        on=[
            "municipio_norm",
            "date"
        ],
        how="left"
    )

    df["inundacao"] = (
        df["inundacao"]
        .fillna(0)
        .astype(int)
    )

    return df


# ============================================================
# GOLD
# ============================================================

def transform_silver_to_gold(
    silver_path,
    atlas_path,
    output_path="data/gold/features_inundacao.parquet"
):
    """
    Silver -> Gold.

    Cria features meteorológicas e temporais
    para estimar a probabilidade de inundação.
    """

    silver_path = Path(silver_path)
    output_path = Path(output_path)

    df = pd.read_parquet(
        silver_path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df = create_flood_target(
        df,
        atlas_path
    )

    # --------------------------------------------------------
    # Features temporais
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "station_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    grouped = df.groupby(
        "station_id"
    )

    # --------------------------------------------------------
    # Precipitação acumulada
    #
    # Importante:
    # são acumulados baseados nos dados diários.
    # --------------------------------------------------------

    df["precip_sum_3d"] = (
        grouped["precip_mm"]
        .transform(
            lambda x: x.rolling(
                3,
                min_periods=1
            ).sum()
        )
    )

    df["precip_sum_7d"] = (
        grouped["precip_mm"]
        .transform(
            lambda x: x.rolling(
                7,
                min_periods=1
            ).sum()
        )
    )

    df["precip_sum_24h"] = (
        df["precip_mm"]
    )

    df["precip_sum_48h"] = (
        grouped["precip_mm"]
        .transform(
            lambda x: x.rolling(
                2,
                min_periods=1
            ).sum()
        )
    )

    df["precip_sum_72h"] = (
        grouped["precip_mm"]
        .transform(
            lambda x: x.rolling(
                3,
                min_periods=1
            ).sum()
        )
    )

    # --------------------------------------------------------
    # Número de dias chuvosos
    # --------------------------------------------------------

    df["rain_day"] = (
        df["precip_mm"] >= 1
    ).astype(int)

    df["rain_days_7d"] = (
        grouped["rain_day"]
        .transform(
            lambda x: x.rolling(
                7,
                min_periods=1
            ).sum()
        )
    )

    # --------------------------------------------------------
    # Dias secos
    # --------------------------------------------------------

    df["dry_day"] = (
        df["precip_mm"] < 1
    ).astype(int)

    df["dry_days_7d"] = (
        grouped["dry_day"]
        .transform(
            lambda x: x.rolling(
                7,
                min_periods=1
            ).sum()
        )
    )

    # --------------------------------------------------------
    # Temperatura
    # --------------------------------------------------------

    df["temperature_avg_3d"] = (
        grouped["temperature_avg"]
        .transform(
            lambda x: x.rolling(
                3,
                min_periods=1
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Umidade
    # --------------------------------------------------------

    df["humidity_avg_3d"] = (
        grouped["humidity_avg"]
        .transform(
            lambda x: x.rolling(
                3,
                min_periods=1
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Pressão
    # --------------------------------------------------------

    df["pressure_change_24h"] = (
        grouped["pressure_avg"]
        .diff()
    )

    # --------------------------------------------------------
    # Sazonalidade
    # --------------------------------------------------------

    day_of_year = (
        df["date"]
        .dt.dayofyear
    )

    df["season_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["season_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # --------------------------------------------------------
    # Features geográficas
    #
    # Coordenadas não são conhecidas pelo processamento
    # somente a partir do nome.
    #
    # Serão mantidas como NaN até que o catálogo oficial
    # de estações seja incorporado.
    # --------------------------------------------------------

    if "latitude" not in df.columns:
        df["latitude"] = np.nan

    if "longitude" not in df.columns:
        df["longitude"] = np.nan

    if "altitude" not in df.columns:
        df["altitude"] = np.nan

    # --------------------------------------------------------
    # Remover colunas auxiliares
    # --------------------------------------------------------

    df = df.drop(
        columns=[
            "municipio_norm",
            "rain_day",
            "dry_day"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "station_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        output_path,
        index=False
    )

    csv_path = output_path.with_suffix(
        ".csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nGold criada:"
        f"\n  {output_path}"
        f"\n  {csv_path}"
        f"\n  Registros: {len(df)}"
        f"\n  Estações: {df['station_id'].nunique()}"
        f"\n  Eventos de inundação: {df['inundacao'].sum()}"
    )

    return df