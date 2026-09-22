"""
Módulo de Ingestão - Camada Bronze
----------------------------------

Responsável por:
- extrair arquivos CSV dos pacotes históricos do INMET;
- preservar os dados brutos;
- trabalhar com todas as estações do Rio Grande do Sul;
- identificar as estações pelo código WMO.

A camada Bronze não tenta transformar ou enriquecer os dados.
O enriquecimento acontece posteriormente na camada Silver.
"""

from pathlib import Path
import zipfile
import shutil


def _is_rs_station_file(filename: str) -> bool:
    """
    Identifica arquivos de estações do Rio Grande do Sul.

    Os arquivos históricos do INMET normalmente seguem o padrão:

    INMET_S_RS_A803_SANTA MARIA_01-01-2025_A_31-12-2025.CSV

    O marcador "_RS_" é utilizado para restringir a ingestão ao RS.
    """

    name = filename.upper()

    return (
        name.endswith(".CSV")
        and "_RS_" in name
        and "INMET_" in name
    )


def ingest_from_inmet_zip(
    zip_path: str,
    output_dir: str = "data/bronze"
):
    """
    Extrai todas as estações do RS presentes no ZIP do INMET.

    Parameters
    ----------
    zip_path : str
        Caminho para o ZIP anual do INMET.

    output_dir : str
        Diretório da camada Bronze.

    Returns
    -------
    list[str]
        Lista dos arquivos extraídos.
    """

    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Arquivo ZIP não encontrado: {zip_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_files = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():

            if member.is_dir():
                continue

            filename = Path(member.filename).name

            if not _is_rs_station_file(filename):
                continue

            destination = output_dir / filename

            with z.open(member) as source, open(
                destination,
                "wb"
            ) as target:
                shutil.copyfileobj(source, target)

            extracted_files.append(str(destination))

            print(
                f"   ✓ Extraído: {filename}"
            )

    print(
        f"\nTotal de arquivos RS extraídos: "
        f"{len(extracted_files)}"
    )

    return extracted_files


def ingest_from_csv(
    csv_path: str,
    output_dir: str = "data/bronze"
):
    """
    Copia um CSV bruto do INMET para a camada Bronze.
    """

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {csv_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = output_dir / csv_path.name

    shutil.copy2(
        csv_path,
        destination
    )

    print(
        f"✓ Arquivo copiado para Bronze: "
        f"{destination}"
    )

    return str(destination)


def list_bronze_stations(
    bronze_dir: str = "data/bronze"
):
    """
    Lista os códigos WMO encontrados na camada Bronze.

    O código é extraído do padrão:

    INMET_S_RS_A803_...
    """

    bronze_dir = Path(bronze_dir)

    if not bronze_dir.exists():
        return []

    stations = set()

    for path in bronze_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() != ".csv":
            continue

        parts = path.name.split("_")

        if len(parts) >= 4:

            station_id = parts[3].strip().upper()

            if station_id:
                stations.add(station_id)

    return sorted(stations)


if __name__ == "__main__":

    import sys

    zip_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if zip_path is None:

        print(
            "Uso:\n"
            "python ingestion.py caminho/para/arquivo.zip"
        )

        raise SystemExit(1)

    extracted = ingest_from_inmet_zip(
        zip_path=zip_path,
        output_dir="data/bronze"
    )

    print(
        f"\n{len(extracted)} arquivos extraídos."
    )

    print(
        "\nEstações encontradas:"
    )

    for station in list_bronze_stations(
        "data/bronze"
    ):
        print(f" - {station}")