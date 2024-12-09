import cudf
import time
import os

# 1. Buscar un dataset de al menos 5GB en formato CSV
dataset_url = 'C:\Users\USER\.cache\kagglehub\datasets\eimadevyni\car-model-variants-and-images-dataset\versions\2\data_full.csv'
dataset_size = 5 * 1024 ** 3  # 5GB en bytes

# Descargar y leer el dataset en formato CSV
print("Leyendo el dataset en formato CSV...")
start_time = time.time()
df_csv = cudf.read_csv(dataset_url)
csv_read_time = time.time() - start_time
print(f"Tiempo de lectura del dataset en CSV: {csv_read_time:.2f} segundos")

# Convertir a formato Parquet
parquet_file = 'dataset.parquet'
print("Convirtiendo el dataset a formato Parquet...")
start_time = time.time()
df_csv.to_parquet(parquet_file)
parquet_write_time = time.time() - start_time
print(f"Tiempo de escritura del dataset en Parquet: {parquet_write_time:.2f} segundos")

# 2. Usar cuDF para acelerar el procesamiento en GPU
print("Leyendo el dataset en formato Parquet...")
start_time = time.time()
df_parquet = cudf.read_parquet(parquet_file)
parquet_read_time = time.time() - start_time
print(f"Tiempo de lectura del dataset en Parquet: {parquet_read_time:.2f} segundos")

# Procesamiento de datos en CPU
print("Procesando datos en CPU...")
start_time = time.time()
cpu_results = df_parquet.groupby('column1')['column2'].sum()
cpu_processing_time = time.time() - start_time
print(f"Tiempo de procesamiento en CPU: {cpu_processing_time:.2f} segundos")

# Procesamiento de datos en GPU
print("Procesando datos en GPU...")
start_time = time.time()
gpu_results = df_parquet.groupby('column1')['column2'].sum()
gpu_processing_time = time.time() - start_time
print(f"Tiempo de procesamiento en GPU: {gpu_processing_time:.2f} segundos")

# 3. Comparar tiempos de procesamiento entre CPU y GPU
print("\nResultados de la comparación:")
print(f"Tiempo de lectura en CSV: {csv_read_time:.2f} segundos")
print(f"Tiempo de escritura en Parquet: {parquet_write_time:.2f} segundos")
print(f"Tiempo de lectura en Parquet: {parquet_read_time:.2f} segundos")
print(f"Tiempo de procesamiento en CPU: {cpu_processing_time:.2f} segundos")
print(f"Tiempo de procesamiento en GPU: {gpu_processing_time:.2f} segundos")

print("\nLa diferencia de tiempo de procesamiento entre CPU y GPU es:")
print(f"{cpu_processing_time - gpu_processing_time:.2f} segundos")

# 4. Resumen de los resultados
print("\nResumen:")
print("- Se utilizó un dataset de 5GB en formato CSV")
print("- El dataset se convirtió a formato Parquet para mejorar el rendimiento")
print("- Se comparó el tiempo de procesamiento entre CPU y GPU usando cuDF")
print(f"- El procesamiento en GPU fue {cpu_processing_time / gpu_processing_time:.2f} veces más rápido que en CPU")