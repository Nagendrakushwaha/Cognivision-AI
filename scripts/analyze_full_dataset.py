import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import io
import json
from PIL import Image

def analyze_full_dataset():
    train_path = "train-00000-of-00001.parquet"
    test_path = "test-00000-of-00001.parquet"

    print("Reading train and test parquet tables...")
    train_table = pq.read_table(train_path)
    test_table = pq.read_table(test_path)
    
    df_train = train_table.to_pandas()
    df_test = test_table.to_pandas()

    print(f"df_train shape: {df_train.shape}")
    print(f"df_test shape: {df_test.shape}")
    print(f"Columns: {df_train.columns.tolist()}")

    # Extract entities details
    print("\n--- Entities Inspection ---")
    train_entities = pd.DataFrame(df_train['entities'].tolist())
    test_entities = pd.DataFrame(df_test['entities'].tolist())
    
    print("Train entities null count:")
    print(train_entities.isnull().sum())
    print("Train entities empty string count:")
    print((train_entities == "").sum())
    
    print("\nTest entities null count:")
    print(test_entities.isnull().sum())
    print("Test entities empty string count:")
    print((test_entities == "").sum())

    print("\nSample train entities (first 3):")
    for i in range(3):
        print(f"Row {i}: {train_entities.iloc[i].to_dict()}")

    # Image sizes inspection
    train_sizes = pd.DataFrame(df_train['image_size'].tolist())
    test_sizes = pd.DataFrame(df_test['image_size'].tolist())
    print("\nTrain Image Sizes Summary:")
    print(train_sizes.describe())

    # Words and Bounding Boxes inspection
    df_train['num_words'] = df_train['words'].apply(lambda w: len(w) if w is not None else 0)
    df_test['num_words'] = df_test['words'].apply(lambda w: len(w) if w is not None else 0)
    df_train['num_bboxes'] = df_train['bboxes'].apply(lambda b: len(b) if b is not None else 0)
    
    print("\nTrain Words count stats:")
    print(df_train['num_words'].describe())

    # Check image bytes decoding
    first_img_data = df_train['image'].iloc[0]
    img = Image.open(io.BytesIO(first_img_data['bytes']))
    print(f"\nDecoded first image: mode={img.mode}, format={img.format}, size={img.size}")
    
    # Check if there are receipt categories or vendor domains or entity patterns
    print("\nTop 10 most common companies in train:")
    print(train_entities['company'].value_counts().head(10))

    # Check words sample & bounding box format
    print("\nWords sample for first receipt:", df_train['words'].iloc[0][:5])
    print("Bboxes sample for first receipt:", df_train['bboxes'].iloc[0][:5])

if __name__ == "__main__":
    analyze_full_dataset()
