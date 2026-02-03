import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "all" # ["1k", "200k", "all"]
CHUNK_SIZE = 100000
GCP_PROJECT = "silicon-sentinel" # TO COMPLETE
GCP_PROJECT_WORKINTECH = "silicon-sentinel"
BQ_DATASET = "mlops"
BQ_REGION = "EU"
MODEL_TARGET = "local"
MIN_DATE = '2009-01-01'
MAX_DATE = '2015-01-01'
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "workintech", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "workintech", "mlops", "training_outputs")

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}

DTYPES_PROCESSED = np.float32

