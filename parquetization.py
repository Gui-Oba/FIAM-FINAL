import duckdb

def parquetize():
  duckdb.sql("""
  COPY (
    SELECT *,
          year(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d'))  AS y,
          month(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS m
    FROM 'ret_sample.csv'
  ) TO 'ret_parquet'
    (FORMAT PARQUET, PARTITION_BY (y,m), COMPRESSION ZSTD);
  """)

if __name__ == "__main__":
  parquetize()