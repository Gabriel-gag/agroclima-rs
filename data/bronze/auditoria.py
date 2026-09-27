from pathlib import Path
import pandas as pd

BRONZE_DIR = Path("data/bronze")

arquivos = sorted(
    p for p in BRONZE_DIR.glob("*.CSV")
    if "_RS_" in p.name.upper()
)

# Pega alguns arquivos que nossa auditoria marcou como problemáticos
problemas = []

for path in arquivos:

    try:
        df = pd.read_csv(
            path,
            sep=";",
            encoding="latin1",
            skiprows=8,
            decimal=",",
            nrows=2
        )

        if "Data" not in df.columns:
            problemas.append(path)

    except Exception:
        pass

print("Arquivos problemáticos encontrados:", len(problemas))

print("\nPrimeiros 10:")
for p in problemas[:10]:
    print(p.name)


path = problemas[0]

print("=" * 80)
print("ARQUIVO:")
print(path.name)
print("=" * 80)

print("\n--- PRIMEIRAS 15 LINHAS BRUTAS ---\n")

with open(path, "r", encoding="latin1") as f:
    for i in range(15):
        linha = f.readline()
        print(f"{i}: {linha.rstrip()}")