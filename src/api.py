"""
API REST de Consulta & Inferência - AgroClima RS

Disponibiliza dados meteorológicos e executa inferência
utilizando o modelo de probabilidade de inundação treinado.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import os
from src.map import criar_mapa

app = FastAPI(
    title="AgroClima RS API",
    description=(
        "API para consulta meteorológica e estimativa da "
        "probabilidade de ocorrência de inundação no Rio Grande do Sul."
    ),
    version="2.1.0"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "../data/gold/features_inundacao.parquet"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.joblib"
)


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

artifact = None

if os.path.exists(MODEL_PATH):
    artifact = joblib.load(MODEL_PATH)


# ============================================================
# FUNÇÃO DE LEITURA DOS DADOS
# ============================================================

def get_data() -> pd.DataFrame:
    """
    Carrega a tabela Gold de features de inundação.
    """

    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=500,
            detail=(
                "Base de dados analítica (Gold) não encontrada: "
                f"{DATA_PATH}"
            )
        )

    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao carregar a base Gold: {exc}"
        )

    return df


# ============================================================
# FUNÇÃO AUXILIAR
# ============================================================

def clean_value(value):
    """
    Converte valores NaN/NaT para None para permitir
    a serialização correta da resposta JSON.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


# ============================================================
# ROTA PRINCIPAL
# ============================================================

@app.get("/")
def root():
    return {
        "projeto": "AgroClima RS",
        "recorte": "Estado do Rio Grande do Sul",
        "objetivo": (
            "Estimativa da probabilidade de ocorrência de inundação"
        ),
        "modelo_carregado": artifact is not None,
        "docs": "/docs"
    }


# ============================================================
# ESTAÇÕES
# ============================================================

@app.get("/estacoes")
def listar_estacoes():
    """
    Retorna o catálogo das estações e municípios monitorados.
    """

    df = get_data()

    colunas = [
        "station_id",
        "codigo_ibge",
        "municipio",
        "latitude",
        "longitude",
        "altitude"
    ]

    colunas_faltantes = [
        coluna
        for coluna in colunas
        if coluna not in df.columns
    ]

    if colunas_faltantes:
        raise HTTPException(
            status_code=500,
            detail=(
                "Colunas necessárias para listar as estações "
                f"não encontradas: {colunas_faltantes}"
            )
        )

    estacoes = (
        df[colunas]
        .drop_duplicates()
        .copy()
    )

    return estacoes.to_dict(orient="records")
# ============================================================
# MAPA
# ============================================================
@app.get(
    "/mapa",
    response_class=HTMLResponse,
    summary="Mapa de probabilidade estimada de inundação"
)
def mapa_inundacao(
    data: str | None = Query(
        default=None,
        description="Data do mapa no formato AAAA-MM-DD"
    )
):

    

    try:

        mapa = criar_mapa(
            data=data
        )

        return mapa.get_root().render()

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar o mapa: {exc}"
        )

# ============================================================
# INFERÊNCIA DE INUNDAÇÃO
# ============================================================

@app.get("/inundacao")
def consultar_inundacao(
    station_id: str = Query(
        ...,
        description="Código WMO da estação (ex: A887, A827, A802)"
    ),
    data: str = Query(
        None,
        description=(
            "Data da observação no formato AAAA-MM-DD "
            "(padrão: última disponível)"
        )
    )
):
    """
    Executa inferência utilizando o modelo treinado.

    Fluxo:
    1. Localiza a estação.
    2. Seleciona a data solicitada ou a última disponível.
    3. Extrai as features utilizadas pelo modelo.
    4. Aplica o imputer.
    5. Executa predict_proba().
    6. Retorna a probabilidade estimada de inundação.
    """

    # --------------------------------------------------------
    # Verifica se o modelo foi carregado
    # --------------------------------------------------------

    if artifact is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Modelo preditivo não encontrado no servidor. "
                "Execute src/train.py."
            )
        )

    # --------------------------------------------------------
    # Valida estrutura do artefato
    # --------------------------------------------------------

    required_artifact_keys = [
        "features",
        "model",
        "imputer",
        "threshold"
    ]

    missing_artifact_keys = [
        key
        for key in required_artifact_keys
        if key not in artifact
    ]

    if missing_artifact_keys:
        raise HTTPException(
            status_code=500,
            detail=(
                "Artefato do modelo incompleto. "
                f"Chaves ausentes: {missing_artifact_keys}"
            )
        )

    # --------------------------------------------------------
    # Carrega os dados
    # --------------------------------------------------------

    df = get_data()

    # --------------------------------------------------------
    # Verifica colunas essenciais
    # --------------------------------------------------------

    required_columns = [
        "station_id",
        "date",
        "codigo_ibge",
        "municipio",
        "precip_mm",
        "precip_sum_3d",
        "precip_sum_7d",
        "precip_sum_24h",
        "precip_sum_48h",
        "precip_sum_72h",
        "temperature_avg",
        "temperature_min",
        "temperature_max",
        "humidity_avg",
        "pressure_avg",
        "pressure_change_24h",
        "wind_speed_avg"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Colunas necessárias não encontradas na base Gold: "
                f"{missing_columns}"
            )
        )

    # --------------------------------------------------------
    # Filtra estação
    # --------------------------------------------------------

    station_id_normalized = station_id.strip().upper()

    filtered = df[
        df["station_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        == station_id_normalized
    ].copy()

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Estação {station_id} não encontrada."
        )

    # --------------------------------------------------------
    # Filtra data
    # --------------------------------------------------------

    filtered["date"] = pd.to_datetime(
        filtered["date"],
        errors="coerce"
    )

    if data:

        requested_date = pd.to_datetime(
            data,
            errors="coerce"
        )

        if pd.isna(requested_date):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Data inválida: {data}. "
                    "Use o formato AAAA-MM-DD."
                )
            )

        row = filtered[
            filtered["date"].dt.date
            == requested_date.date()
        ]

        if row.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Data {data} não encontrada "
                    f"para a estação {station_id}."
                )
            )

        record = row.iloc[0]

    else:

        filtered = filtered.dropna(
            subset=["date"]
        )

        if filtered.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Nenhuma data válida encontrada "
                    f"para a estação {station_id}."
                )
            )

        record = (
            filtered
            .sort_values("date")
            .iloc[-1]
        )

    # --------------------------------------------------------
    # Recupera configurações do artefato
    # --------------------------------------------------------

    features = artifact["features"]
    model = artifact["model"]
    imputer = artifact["imputer"]
    threshold = artifact["threshold"]

    model_type = artifact.get(
        "model_type",
        type(model).__name__
    )

    # --------------------------------------------------------
    # Verifica se todas as features existem
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in record.index
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "Features necessárias para o modelo "
                f"não encontradas na base Gold: {missing_features}"
            )
        )

    # --------------------------------------------------------
    # Prepara entrada do modelo
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [record[features]]
    )

    try:
        input_imputed = imputer.transform(
            input_df
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao aplicar o imputer às features: "
                f"{exc}"
            )
        )

    # --------------------------------------------------------
    # Inferência
    # --------------------------------------------------------

    try:
        prob_inundacao = float(
            model.predict_proba(
                input_imputed
            )[0, 1]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao executar a inferência do modelo: "
                f"{exc}"
            )
        )

    inundacao_prevista = (
        prob_inundacao >= threshold
    )

    # --------------------------------------------------------
    # Código IBGE
    # --------------------------------------------------------

    codigo_ibge = record["codigo_ibge"]

    if pd.isna(codigo_ibge):
        codigo_ibge = None
    else:
        codigo_ibge = int(codigo_ibge)

    # --------------------------------------------------------
    # Resposta
    # --------------------------------------------------------

    return {
        "station_id": clean_value(
            record["station_id"]
        ),

        "codigo_ibge": codigo_ibge,

        "municipio": clean_value(
            record["municipio"]
        ),

        "data_observacao": (
            record["date"].strftime("%Y-%m-%d")
            if not pd.isna(record["date"])
            else None
        ),

        "condicoes_observadas": {

            "chuva_mm": clean_value(
                record.get("precip_mm")
            ),

            "chuva_acumulada_3d_mm": clean_value(
                record.get("precip_sum_3d")
            ),

            "chuva_acumulada_7d_mm": clean_value(
                record.get("precip_sum_7d")
            ),

            "chuva_acumulada_24h_mm": clean_value(
                record.get("precip_sum_24h")
            ),

            "chuva_acumulada_48h_mm": clean_value(
                record.get("precip_sum_48h")
            ),

            "chuva_acumulada_72h_mm": clean_value(
                record.get("precip_sum_72h")
            ),

            "temperatura_media_c": clean_value(
                record.get("temperature_avg")
            ),

            "temperatura_min_c": clean_value(
                record.get("temperature_min")
            ),

            "temperatura_max_c": clean_value(
                record.get("temperature_max")
            ),

            "umidade_media_pct": clean_value(
                record.get("humidity_avg")
            ),

            "pressao_hpa": clean_value(
                record.get("pressure_avg")
            ),

            "variacao_pressao_24h": clean_value(
                record.get("pressure_change_24h")
            ),

            "velocidade_vento": clean_value(
                record.get("wind_speed_avg")
            )
        },

        "inferencia_modelo": {

            "modelo": model_type,

            "probabilidade_inundacao": round(
                prob_inundacao,
                4
            ),

            "probabilidade_percentual": round(
                prob_inundacao * 100,
                2
            ),

            "threshold_decisao": threshold,

            "inundacao_prevista": inundacao_prevista,

            "status": (
                "Probabilidade elevada"
                if inundacao_prevista
                else "Probabilidade abaixo do threshold"
            )
        }
    }