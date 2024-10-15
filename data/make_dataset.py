import os
import chardet
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

# Путь к корневой папке
root_dir = 'PowerShellCorpus'
output_parquet_path = './data.parquet'

# Инициализация Parquet writer
schema = pa.schema([
    ('path', pa.string()),
    ('content', pa.string()),
    ('class', pa.int32())
])
writer = pq.ParquetWriter(output_parquet_path, schema)

max_folders = 100000
folder_count = 0
confidence_threshold = 0.9


# Список допустимых папок для обработки и корневая папка для них
allowed_folders = ["GithubGist", "InvokeCradleCrafter", "InvokeObfuscation", "IseSteroids", "PoshCode"]
allowed_folders_full = [os.path.join(root_dir, folder) for folder in allowed_folders]

print("Processing started...")
start_time = time.time()

for root, dirs, files in os.walk(root_dir):
    if any(root.startswith(allowed_path) for allowed_path in allowed_folders_full):
        print(f"Processing folder: {root}")
        folder_start_time = time.time()

        processed_files = 0
        for file in files:
            if file.endswith('.ps1') or file.endswith('.psm1'):
                print(f"Reading file: {file}")
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    raw_data = f.read(5000)
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    confidence = result['confidence']

                if confidence < confidence_threshold:
                    print("Skipped due to low confidence")

                    continue  # Пропуск файла из-за низкой уверенности в кодировке

                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()

                script_class = 1 if 'InvokeObfuscation' in root else 0
                data = {
                    'path': file_path,
                    'content': content,
                    'class': script_class
                }
                print(f"Adding data for file: {file}")

                # Создание таблицы из данных и запись в Parquet
                table = pa.Table.from_pandas(pd.DataFrame([data]), schema=schema, preserve_index=False)
                writer.write_table(table)
                processed_files += 1

        folder_end_time = time.time()
        print(
            f"Finished processing {root}. Files processed: {processed_files}. Time taken: {folder_end_time - folder_start_time:.2f} seconds.")

        folder_count += 1
        # if folder_count >= max_folders:
        #     break

writer.close()
end_time = time.time()
print(f"Data has been written to Parquet file. Total time: {end_time - start_time:.2f} seconds.")
