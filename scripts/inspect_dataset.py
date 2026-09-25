import os
import io
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image

def inspect():
    train_path = "train-00000-of-00001.parquet"
    test_path = "test-00000-of-00001.parquet"
    
    print("=== INSPECTING PARQUET METADATA ===")
    for name, path in [("TRAIN", train_path), ("TEST", test_path)]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow
        print(f"\n--- {name} SET: {path} ---")
        print(f"File Size: {file_size_mb:.2f} MB")
        print(f"Number of rows: {metadata.num_rows}")
        print(f"Number of columns: {metadata.num_columns}")
        print(f"Number of row groups: {metadata.num_row_groups}")
        print("Schema fields:")
        for field in schema:
            print(f"  - {field.name}: {field.type}")
            
    print("\n=== INSPECTING FIRST 5 ROWS OF TRAIN ===")
    train_pf = pq.ParquetFile(train_path)
    # Read first row group or small batch
    first_rg = train_pf.read_row_group(0)
    df_sample = first_rg.slice(0, 5).to_pandas()
    print("Columns:", df_sample.columns.tolist())
    for col in df_sample.columns:
        val = df_sample[col].iloc[0]
        val_type = type(val)
        print(f"\nCol '{col}': type={val_type}")
        if isinstance(val, dict):
            print("  Dict keys:", list(val.keys()))
            for k, v in val.items():
                print(f"    {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        elif isinstance(val, bytes):
            print(f"  Bytes length: {len(val)}")
        else:
            print(f"  Sample value: {val}")

    # Inspect class/label distribution across train and test
    print("\n=== LABEL & CLASS DISTRIBUTION ===")
    for name, path in [("TRAIN", train_path), ("TEST", test_path)]:
        table = pq.read_table(path, columns=[c for c in ['label', 'labels', 'category', 'class', 'target'] if c in train_pf.schema_arrow.names])
        df_labels = table.to_pandas()
        print(f"\n{name} Label Summary:")
        print(df_labels.describe(include='all'))
        for col in df_labels.columns:
            vc = df_labels[col].value_counts().sort_index()
            print(f"Value counts for {col} (total {len(vc)} unique):")
            print(vc)

    # Inspect sample image from train if present
    print("\n=== IMAGE DATA VERIFICATION ===")
    for col in df_sample.columns:
        val = df_sample[col].iloc[0]
        img_bytes = None
        if isinstance(val, dict) and 'bytes' in val:
            img_bytes = val['bytes']
            print(f"Image found in dict col '{col}'['bytes']")
        elif isinstance(val, bytes):
            img_bytes = val
            print(f"Image found in bytes col '{col}'")
        
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                print(f"  Image successfully decoded: format={img.format}, size={img.size} (WxH), mode={img.mode}")
            except Exception as e:
                print(f"  Failed to decode image bytes: {e}")

if __name__ == "__main__":
    inspect()
