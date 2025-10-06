from pathlib import Path
import duckdb

SOURCE_DIR = Path("lemonlime")
OUTPUT_CSV = Path("predictions2.csv")

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR.resolve()}")

pattern = str(SOURCE_DIR / "**" / "*.parquet")

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
