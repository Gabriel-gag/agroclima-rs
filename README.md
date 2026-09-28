#AgroClima RS

> Pipeline de dados meteorológicos e estimativa de probabilidade de inundação para municípios do Rio Grande do Sul.

O **AgroClima RS** combina dados meteorológicos do **INMET**, registros históricos de desastres, dados geográficos dos municípios e previsões meteorológicas do **Open-Meteo** para construir um pipeline de dados e um modelo de Machine Learning capaz de estimar a probabilidade de ocorrência de **inundação**.

O projeto também gera um **mapa interativo do Rio Grande do Sul**, permitindo consultar o histórico e visualizar previsões para os próximos **16 dias**.

---

## Objetivo

O projeto busca responder:

1. Como as condições meteorológicas se relacionam com a ocorrência de inundações no histórico?
2. Dadas as condições meteorológicas previstas, qual é a probabilidade estimada de inundação nos próximos dias?

O resultado é uma estimativa probabilística para análise exploratória e visualização de risco.

> **Importante:** a saída do modelo não representa uma previsão determinística de que uma inundação ocorrerá. Trata-se de uma **probabilidade estimada pelo modelo**, condicionada aos dados e às hipóteses utilizadas.

---

## Principais resultados

O projeto produz:

- probabilidades estimadas de inundação por estação;
- consolidação das probabilidades por município usando o código IBGE;
- previsão para 16 dias;
- ranking dos municípios pela probabilidade acumulada;
- mapa interativo com histórico e previsão;
- painel com o Top 5 dos municípios por probabilidade acumulada.

Na execução mais recente do pipeline de previsão:

- **100 estações** processadas;
- **1.600 previsões** geradas;
- **88 municípios** associados a códigos IBGE válidos;
- **16 dias de previsão por estação**.


---

## Arquitetura

O pipeline utiliza uma organização inspirada na arquitetura Medallion:

    INMET / Atlas
          │
          ▼
       BRONZE
    Dados brutos
          │
          ▼
       SILVER
    Dados diários
          │
          ▼
        GOLD
    Features + target
          │
          ├──────────────────┐
          ▼                  ▼
    Machine Learning     Open-Meteo
    Modelo calibrado    Previsão 16d
          │                  │
          └────────┬─────────┘
                   ▼
          Probabilidades de
             inundação
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Ranking             Mapa
      municipal         interativo

---

## Definição do problema

O alvo utilizado pelo modelo é:

    inundacao_t1 = ocorrência de inundação no dia seguinte

As variáveis meteorológicas do dia t são utilizadas para estimar a ocorrência no dia t+1.

Essa formulação D+1 permite separar as variáveis de entrada do evento que está sendo previsto.

---

## Modelo de Machine Learning

O modelo utilizado para gerar as probabilidades operacionais é uma **Regressão Logística calibrada**.

Divisão temporal:

| Período | Uso |
|---|---|
| 2006–2019 | Treinamento |
| 2020–2023 | Calibração |
| 2024–2025 | Holdout temporal |

A calibração utiliza o método **sigmoid** do Scikit-learn.

O artefato utilizado na previsão é:

    src/model_calibrado.joblib

O artefato armazena as features e o modelo calibrado.

### Principais features

**Precipitação**
- precipitação diária;
- acumulados de 24h, 48h e 72h;
- acumulados de 3 e 7 dias;
- dias com chuva nos últimos 7 dias;
- dias secos nos últimos 7 dias.

**Condições atmosféricas**
- temperatura mínima, máxima e média;
- média móvel da temperatura;
- umidade média;
- média móvel da umidade;
- pressão atmosférica;
- variação da pressão em 24h;
- velocidade média do vento;
- radiação solar.

**Sazonalidade e localização**
- seno e cosseno do dia do ano;
- altitude;
- latitude;
- longitude.

---

## Avaliação

A avaliação principal utiliza um **holdout temporal de 2024–2025**, separado do período de treinamento e calibração.

São calculadas:

- **PR-AUC**;
- **ROC-AUC**;
- **Brier Score**.

O uso de PR-AUC é especialmente relevante devido ao forte desbalanceamento entre eventos de inundação e não-eventos.


---

## Previsão meteorológica

A previsão futura utiliza a API do **Open-Meteo**.

Para cada estação são obtidos:

- 7 dias de contexto histórico recente;
- 16 dias de previsão.

Os dados horários são agregados para escala diária e transformados nas mesmas features esperadas pelo modelo histórico.

Variáveis meteorológicas utilizadas:

- temperatura;
- umidade relativa;
- precipitação;
- probabilidade de precipitação;
- pressão atmosférica;
- velocidade do vento;
- rajadas;
- radiação solar.

---

## Probabilidade acumulada em 16 dias

As estações são consolidadas por código IBGE.

Quando existem várias estações associadas ao mesmo município, o pipeline utiliza a **maior probabilidade diária** entre elas.

A probabilidade de pelo menos uma ocorrência durante os 16 dias é calculada por:

    P(acumulada) = 1 - produto(1 - p_d)

onde p_d é a probabilidade estimada para cada dia.

### Hipótese

O cálculo assume **independência entre os dias**. Portanto, a probabilidade acumulada é uma estimativa derivada do modelo sob essa hipótese, e não uma probabilidade observacional direta.

---

## Mapa interativo

O arquivo **src/map.py** gera um mapa utilizando **Folium**.

### Histórico

Permite consultar a probabilidade estimada para uma data disponível na base histórica, juntamente com:

- município;
- código IBGE;
- probabilidade estimada para D+1;
- chuva acumulada em 24h, 72h e 7 dias;
- temperatura média;
- umidade;
- pressão;
- variação da pressão em 24h.

### Previsão

Permite selecionar:

- D+1;
- D+2;
- ...
- D+16.

Também apresenta o **Top 5 municipal** pela probabilidade acumulada no horizonte de 16 dias.

O arquivo gerado é:

    mapa_inundacao_rs.html

Como o HTML incorpora o conteúdo geográfico necessário para o mapa, seu tamanho é elevado. Ele deve ser tratado como um **artefato gerado**, e não como a principal fonte de reprodução do projeto.

---

## Estrutura do repositório

    agroclima-rs/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── data/
    │   ├── atlas/
    │   ├── bronze/
    │   ├── silver/
    │   ├── gold/
    │   ├── forecast/
    │   └── geo/
    │
    ├── notebook/
    │   └── AgroClima_RS_Pelotas_4302_aprimorado.ipynb
    │
    ├── src/
    │   ├── ingestion.py
    │   ├── transformation.py
    │   ├── pipeline.py
    │   ├── modeling.py
    │   ├── train.py
    │   ├── calibrar.py
    │   ├── model_calibrado.joblib
    │   ├── forecast.py
    │   ├── forecast_features.py
    │   ├── predict_forecast.py
    │   ├── gerar_previsao_rs.py
    │   ├── raking_previsao.py
    │   ├── create_geojson.py
    │   └── map.py
    │
    ├── tests/
    │   └── test_pipeline.py
    │
    └── terraform/


> O fluxo principal da versão atual utiliza **model_calibrado.joblib**, **gerar_previsao_rs.py**, **raking_previsao.py** e **map.py**. Alguns scripts presentes no repositório correspondem a etapas ou experimentos anteriores.

---

## Como executar

### 1. Instalar dependências

    python -m pip install -r requirements.txt

### 2. Executar os testes

    python -m unittest discover -s tests

### 3. Gerar as previsões de 16 dias

Com o dataset Gold disponível:

    python src/gerar_previsao_rs.py

Saída:

    data/forecast/probabilidades_inundacao_16d.parquet

### 4. Gerar o ranking municipal

    python src/raking_previsao.py

Saída:

    data/forecast/ranking_municipios_16d.parquet

### 5. Gerar o mapa

    python src/map.py

Saída:

    mapa_inundacao_rs.html

---

## Testes

Os testes automatizados estão em:

    tests/test_pipeline.py

Execute com:

    python -m unittest discover -s tests

---

## Dados

### INMET

Fornece as séries meteorológicas históricas das estações utilizadas no pipeline.

### Atlas de Desastres

Fornece os registros históricos de desastres utilizados para construir o target de inundação.

### Open-Meteo

Fornece as condições meteorológicas previstas utilizadas na etapa operacional de previsão.

### Dados geográficos

Os dados municipais do Rio Grande do Sul são utilizados para relacionar as probabilidades aos códigos IBGE e gerar o mapa.

---

## Limitações

### Eventos raros

Inundações representam uma parcela muito pequena das observações. O problema é, portanto, altamente desbalanceado.

### Probabilidade estimada ≠ risco real

A probabilidade produzida pelo modelo representa a relação aprendida entre as variáveis disponíveis e os eventos históricos.

Uma avaliação completa de risco deveria incorporar também:

- exposição populacional;
- vulnerabilidade;
- uso e cobertura do solo;
- relevo e declividade;
- proximidade de rios;
- características da drenagem;
- infraestrutura urbana.

Assim, o projeto deve ser entendido como um **modelo de probabilidade associada à ocorrência de inundação**, e não como uma avaliação completa de risco socioambiental.

### Dependência entre dias

A probabilidade acumulada de 16 dias assume independência entre as probabilidades diárias. Eventos meteorológicos reais possuem dependência temporal.

### Previsões meteorológicas

As previsões do Open-Meteo são atualizadas continuamente. Uma nova execução pode produzir probabilidades diferentes para o mesmo período futuro.

### Dados

A cobertura histórica e a qualidade das observações variam entre estações e períodos.

---

## Próximos passos

- incorporar declividade e distância a rios;
- incluir população e exposição territorial;
- comparar Random Forest, XGBoost e outros modelos;
- melhorar a calibração probabilística;
- utilizar validação temporal com janelas móveis;
- estudar dependência temporal entre probabilidades diárias;
- automatizar a atualização das previsões;
- disponibilizar o mapa através de uma aplicação web;
- reduzir o tamanho dos artefatos geográficos e dos dados versionados;

---

## Tecnologias

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **PyArrow / Parquet**
- **Requests**
- **Folium**
- **FastAPI**
- **Matplotlib**
- **Seaborn**
- **Terraform / AWS**

---

## Fontes

- [INMET — Instituto Nacional de Meteorologia](https://portal.inmet.gov.br/)
- [Open-Meteo](https://open-meteo.com/)
- [Atlas Brasileiro de Desastres](https://s2id.mi.gov.br/)
- [IBGE](https://www.ibge.gov.br/)

---

## Licença

Este projeto é disponibilizado sob a licença [MIT](https://opensource.org/licenses/MIT).
