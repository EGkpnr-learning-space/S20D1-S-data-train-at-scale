import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from taxifare.params import (
    GCP_PROJECT,
    GCP_PROJECT_WORKINTECH,
    BQ_DATASET,
    DATA_SIZE,
    COLUMN_NAMES_RAW,
    DTYPES_RAW,
    MIN_DATE,  # <--- Bunu ekle
    MAX_DATE,  # <--- Bunu ekle
    LOCAL_DATA_PATH
)
from taxifare.params import *
from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, save_results, load_model
from taxifare.ml_logic.model import compile_model, initialize_model, train_model

def preprocess_and_train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    


    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WORKINTECH}.{BQ_DATASET}.{BQ_DATASET}_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{MIN_DATE}' AND '{MAX_DATE}'
        ORDER BY pickup_datetime
    """
    print(query)    
    
    # Retrieve `query` data from BigQuery or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")

        # HATA DÜZELTME KODU:
        # Pandas read_csv, dtype parametresi içinde timezone-aware datetime kabul etmez.
        # Bu yüzden DTYPES_RAW içinden datetime olanları geçici olarak siliyoruz.
        dtypes_for_loading = DTYPES_RAW.copy()
        if "pickup_datetime" in dtypes_for_loading:
            del dtypes_for_loading["pickup_datetime"]
        if "key" in dtypes_for_loading:
            del dtypes_for_loading["key"]

        data = pd.read_csv(
            data_query_cache_path,
            parse_dates=['pickup_datetime'], # Tarih dönüşümünü burası yapar
            dtype=dtypes_for_loading         # Tarih olmayan diğer tipler (float32 vb.) buradan gelir
        )

    else:
        print("Loading data from Querying Big Query server...")

        # BigQuery bağlantısını kur ve sorguyu çalıştır
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()

        # Save it locally to accelerate the next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    # Clean data using data.py
    if data is None: # Hata yönetimi için opsiyonel kontrol
        print("❌ No data loaded")
        return None

    # Yukarıda tanımladığımız clean_data fonksiyonunu çağırıyoruz
    data = clean_data(data)

    # Create (X_train, y_train, X_val, y_val) without data leaks
    # No need for test sets, we'll report val metrics only
    split_ratio = 0.02 # About one month of validation data


    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: we can `fit_transform` both X_train and X_val without data leakage!
    # 1. Kronolojik Ayırma (Chronological Split)
    # Verinin son %2'lik kısmını doğrulama (validation) için ayırıyoruz.
    train_length = int(len(data) * (1 - split_ratio))
    
    df_train = data.iloc[:train_length, :]
    df_val = data.iloc[train_length:, :]

    # 2. X ve y Ayrımı
    # Hedef değişkenimiz 'fare_amount'
    X_train = df_train.drop("fare_amount", axis=1)
    y_train = df_train[["fare_amount"]]

    X_val = df_val.drop("fare_amount", axis=1)
    y_val = df_val[["fare_amount"]]

    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: we can `fit_transform` both X_train and X_val without data leakage!
    
    # 3. Ön İşleme (Preprocessing)
    # Bu fonksiyonun 'taxifare.ml_logic.preprocessor' modülünden import edildiğini varsayıyoruz.
    # Notebook'taki 'final_preprocessor' mantığını içeren fonksiyondur.
    from taxifare.ml_logic.preprocessor import preprocess_features

    X_train_processed = preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)

    print("✅ data processed")

    # Train a model on the training set, using `model.py`
    model = None
    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    # 1. Modeli Başlat
    model = initialize_model(input_shape=X_train_processed.shape[1:])
    
    # 2. Modeli Derle (Compile)
    model = compile_model(model, learning_rate=learning_rate)
    
    # 3. Modeli Eğit (Train)
    # X_val_processed ve y_val değişkenlerinin bir önceki adımda oluşturulduğunu varsayıyoruz.
    model, history = train_model(
        model, 
        X_train_processed, 
        y_train, 
        batch_size=batch_size, 
        patience=patience, 
        validation_data=(X_val_processed, y_val)
    )

    # Compute the validation metric (min val_mae) of the holdout set
    val_mae = np.min(history.history['val_mae'])

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        # preprocess()
        # train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    Query and preprocess the raw dataset iteratively (by chunks).
    Then store the newly processed (and raw) data on local hard-drive for later re-use.
    - If raw data already exists on local disk:
        - use `pd.read_csv(..., chunksize=CHUNK_SIZE)`
    - If raw data does not yet exists:
        - use `bigquery.Client().query().result().to_dataframe_iterable()`
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features
    from dateutil.parser import parse # Tarih ayrıştırma için gerekli olabilir

    # Tarihleri string formatına çevir
    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    # params.py dosyasındaki değişkenleri kullandığımızdan emin olalım
    # NOT: params.py içinde DATA_SIZE="all" yaptığınızda tablo adı değişebilir, 
    # ancak burada snippet'teki mantığı koruyoruz.
    
    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WORKINTECH}`.{BQ_DATASET}.{BQ_DATASET}_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    
    # Retrieve `query` data as dataframe iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    
    # Eğer daha önce oluşturulmuş processed dosyası varsa silelim (temiz başlangıç)
    if data_processed_path.is_file():
        data_processed_path.unlink()

    if data_query_cache_exists:
        print("Get a dataframe iterable from local CSV...")
        
        # Datetime hatasını önlemek için dtype düzeltmesi
        dtypes_loading = DTYPES_RAW.copy()
        if "pickup_datetime" in dtypes_loading:
            del dtypes_loading["pickup_datetime"]
            
        chunks = pd.read_csv(
            data_query_cache_path,
            chunksize=CHUNK_SIZE, # Bu parametre, veriyi parça parça okumamızı sağlar
            parse_dates=['pickup_datetime'],
            dtype=dtypes_loading
        )

    else:
        print("Get a dataframe iterable from Querying Big Query server...")
        
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result(page_size=CHUNK_SIZE) # BigQuery sonucunu sayfalandır
        chunks = result.to_dataframe_iterable() # Sonucu yinelenebilir (iterable) hale getir

    for chunk_id, chunk in enumerate(chunks):
        print(f"processing chunk {chunk_id}...")

        # 1. Clean chunk
        chunk_clean = clean_data(chunk)

        # 2. Create chunk_processed
        # X ve y ayrımı
        X_chunk = chunk_clean.drop("fare_amount", axis=1)
        y_chunk = chunk_clean[["fare_amount"]]
        
        # Özellikleri işle (stateless preprocessor sayesinde)
        X_processed_chunk = preprocess_features(X_chunk)
        
        # İşlenmiş X (numpy array) ile y (dataframe) birleştir
        # X_processed_chunk bir numpy array olduğu için DataFrame'e çevirip birleştirmek daha güvenlidir
        chunk_processed = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1)
        )

        # 3. Save and append the processed chunk to a local CSV
        # mode='a' (append) ile dosyanın sonuna ekle
        # header=False (sütun isimlerini tekrar tekrar yazmamak için)
        # index=False (gereksiz index numaralarını yazmamak için)
        chunk_processed.to_csv(
            data_processed_path,
            mode='a',
            header=False,
            index=False
        )

        # 4. Save and append the raw chunk if not `data_query_cache_exists`
        # Eğer veri BigQuery'den geldiyse, ham halini de cache'leyelim
        if not data_query_cache_exists:
            # Sadece ilk chunk'ta (chunk_id == 0) başlık (header) yazılır
            is_first_chunk = (chunk_id == 0)
            chunk.to_csv(
                data_query_cache_path,
                mode='a',
                header=is_first_chunk,
                index=False
            )
            
    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")
    
    

def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental train on the (already preprocessed) dataset locally stored.
    - Loading data chunk-by-chunk
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunks, and final model weights on local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: train by batch" + Style.RESET_ALL)
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    # Params.py'dan gelen değişkenlerin import edildiğinden emin olunmalı,
    # ancak fonksiyon içinde zaten scope dahilindelerse sorun yok.
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store each val_mae of each chunk

    # CSV dosyasını okurken dtype belirtmek bellek optimizasyonu için önemlidir.
    # DTYPES_PROCESSED params.py dosyasında tanımlı olmalıdır (float32 vb.)
    chunks = pd.read_csv(data_processed_path,
                         chunksize=CHUNK_SIZE,
                         header=None,
                         dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        print(f"training on preprocessed chunk n°{chunk_id}")
        
        # Her parça için eğitim parametreleri
        learning_rate = 0.0005
        batch_size = 256
        patience = 2
        split_ratio = 0.1 # Parçalar küçük olduğu için %10 validation ayırmak mantıklı

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        
        # Veriyi karıştır (shuffle) ve numpy array'e çevir
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        # Özellikler (X) ve Hedef (y) ayrımı
        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # --- YOUR CODE HERE BAŞLANGICI ---
        
        # 1. Modeli Başlatma (Sadece ilk parçada yapılır!)
        if model is None:
            # Model henüz yoksa, ilk parçanın boyutuna göre başlat ve derle
            model = initialize_model(input_shape=X_train_chunk.shape[1:])
            model = compile_model(model, learning_rate=learning_rate)
        
        # 2. Modeli Eğitme (Incremental Learning)
        # Keras modelleri, tekrar .fit() çağrıldığında ağırlıklarını korur (resetlenmez).
        # Bu sayede her chunk ile model biraz daha öğrenir.
        model, history = train_model(
            model, 
            X_train_chunk, 
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_chunk, y_val_chunk)
        )
        
        # 3. Metrikleri Kaydetme
        # O anki chunk'ın en iyi validation MAE değerini listeye ekle
        val_mae = np.min(history.history['val_mae'])
        metrics_val_list.append(val_mae)
        
        # --- YOUR CODE HERE BİTİŞİ ---

    # Return the last value of the validation MAE
    # (veya tüm chunkların ortalaması da alınabilir ama son durum genelde en iyisidir)
    if len(metrics_val_list) == 0:
        print("❌ No chunks were trained!")
        return None

    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

    # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    from taxifare.ml_logic.registry import load_model
    from taxifare.ml_logic.preprocessor import preprocess_features

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")
    return y_pred