from parquetization import parquetize
from preprocess_parquets import preprocess
from feature_select import feature_select
from model import model
from parquet_to_csv import parquet_to_csv
from portfolio_optimization import portfolio

def main():
    parquetize()
    preprocess()
    feature_select()
    model()
    parquet_to_csv()
    portfolio()

if __name__ == "__main__":
    main()