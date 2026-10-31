
import pandas as pd
from pathlib import Path

data_dir = Path("data/raw")

files = [
    data_dir / "student-mat.csv",
    data_dir / "student-por.csv",
]

for file in files:
    print(f"\n{'=' * 50}")
    print(f"Dataset: {file.name}")

    if not file.exists():
        print("FILE NOT FOUND — check the file location.")
        continue

    df = pd.read_csv(file, sep=";")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:", df.isnull().sum().sum())
    print("Final grade (G3) distribution:")
    print(df["G3"].value_counts().sort_index())
