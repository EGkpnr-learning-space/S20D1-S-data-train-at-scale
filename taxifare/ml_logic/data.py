import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *
from taxifare.utils import simple_time_and_memory_tracker


@simple_time_and_memory_tracker
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # 1. Compress raw_data by setting types to DTYPES_RAW
    # Bu DTYPES_RAW değişkeninin params.py'dan geldiğini varsayıyoruz.
    df = df.astype(DTYPES_RAW)

    # 2. Remove buggy transactions
    df = df.drop_duplicates()  # Tekrar eden satırları sil
    df = df.dropna(how='any', axis=0) # Eksik veri olan satırları sil
    
    # Mantıksız değerleri temizle (Negatif yolcu veya ücret gibi)
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    
    if "fare_amount" in df.columns:
        df = df[df.fare_amount.between(0, 400)]
        
    if "passenger_count" in df.columns:
        df = df[df.passenger_count.between(0, 8)]

    # 3. Remove geographically irrelevant transactions (rows)
    # Notebook'taki bounding box değerleri: 40.5 - 40.9 ve -74.3 - -73.7
    df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    print("✅ data cleaned")

    return df