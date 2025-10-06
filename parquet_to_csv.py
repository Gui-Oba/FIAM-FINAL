from pathlib import Path
import duckdb

def parquet_to_csv():
    SOURCE_DIR = Path("oos_preds_pca")   # was ""
    pattern = str(SOURCE_DIR / "**" / "*.parquet")
    OUTPUT_CSV = Path("predictions2.csv")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR.resolve()}")

    print(f"Reading parquet files matching: {pattern}")
    print(f"Writing combined CSV to: {OUTPUT_CSV.resolve()}")

    query = f"""
    COPY (
        SELECT *
        FROM read_parquet('{pattern}')
    ) TO '{OUTPUT_CSV}' WITH (HEADER, DELIMITER ',');
    """

    duckdb.query(query)
    print("Export complete.")

if __name__ == "__main__":
    parquet_to_csv()