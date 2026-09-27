from forecast import (
    buscar_previsao,
    agregar_previsao_diaria,
)

from forecast_features import (
    preparar_features_previsao,
)


STATION_ID = "A803"

LATITUDE = -29.725
LONGITUDE = -53.7206
ALTITUDE = 103.1


print("=" * 60)
print("TESTE DE PREVISÃO")
print("=" * 60)


# ==========================================
# 1. Buscar histórico recente + previsão
# ==========================================

df_horario = buscar_previsao(
    latitude=LATITUDE,
    longitude=LONGITUDE,
    dias=16,
    dias_historico=7,
)

print("\nDados horários:")
print(df_horario.shape)

print(
    df_horario["datetime"].min(),
    "->",
    df_horario["datetime"].max(),
)


# ==========================================
# 2. Agregar por dia
# ==========================================

df_diario = agregar_previsao_diaria(
    df_horario
)

print("\nDados diários:")
print(df_diario.shape)

print(
    df_diario[
        [
            "date",
            "precip_mm",
            "temperature_avg",
            "pressure_avg",
        ]
    ].to_string(index=False)
)


# ==========================================
# 3. Criar features
# ==========================================

df_features = preparar_features_previsao(
    df_diario,
    station_id=STATION_ID,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    altitude=ALTITUDE,
)


print("\nFeatures finais:")
print(df_features.shape)

print(
    df_features[
        [
            "date",
            "precip_mm",
            "precip_sum_24h",
            "precip_sum_48h",
            "precip_sum_72h",
            "precip_sum_7d",
            "temperature_avg_3d",
            "pressure_change_24h",
        ]
    ].to_string(index=False)
)


# ==========================================
# 4. Verificar NaN
# ==========================================

print("\nValores ausentes:")

print(
    df_features.isna()
    .sum()
    .loc[
        lambda x: x > 0
    ]
)


# ==========================================
# 5. Verificar quantidade
# ==========================================

print("\nQuantidade de dias previstos:")

print(
    len(df_features)
)