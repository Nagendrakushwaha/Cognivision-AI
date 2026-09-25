import os
import io
import time
import json
import torch
import pyarrow.parquet as pq
from PIL import Image
import torchvision.transforms as transforms

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def categorize_receipt(company_name, words):
    c = (company_name or '').upper()
    all_words_text = " ".join([str(w).upper() for w in (words if words is not None else [])])
    full_text = f"{c} {all_words_text}"

    if any(k in full_text for k in ["MR. D.I.Y.", "MR D.I.Y", "MR DIY", "HARDWARE", "PAPAN", "TIMBER", "BUILDING", "HOME DECO", "AIK HUAT", "ELECTRICAL"]):
        return "Hardware & Home"
    if any(k in full_text for k in ["GARDENIA", "BAKERY", "BAKERIES", "CONFECTIONERY", "CAKE", "BREAD", "ROTI", "PASTRY"]):
        return "Bakery & Confectionery"
    if any(k in full_text for k in ["SPEED MART", "SPEEDMART", "99 SPEED", "AEON", "GROCER", "SUPERMARKET", "PASARAYA", "HYPERMARKET", "MINI MARKET", "MART"]):
        return "Supermarket & Grocery"
    if any(k in full_text for k in ["RESTORAN", "RESTAURANT", "FOOD", "KOPITIAM", "CAFE", "COFFEE", "SEAFOOD", "UNIHAKKA", "GERBANG ALAF", "STEAMBOAT", "KITCHEN", "DINING", "MCDONALD"]):
        return "Restaurant & Dining"
    if any(k in full_text for k in ["STATIONERY", "BOOK", "POPULAR BOOK", "PAPER", "PEN", "OFFICE SUPPLIES", "PRINTING"]):
        return "Stationery & Bookstore"
    return "General Retail & Services"

def build_cache():
    os.makedirs("artifacts/cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() # float32 in [0, 1]
    ])

    for split_name, parquet_file in [("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        cache_file = f"artifacts/cache/{split_name}_cache.pt"
        if os.path.exists(cache_file):
            print(f"Cache already exists: {cache_file}")
            continue

        print(f"\nBuilding cache for {split_name} from {parquet_file}...")
        start_t = time.time()
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        tensors = []
        labels = []
        keys = []
        records_meta = []
        
        for idx, row in df.iterrows():
            key = row['key']
            company = row['entities'].get('company') if row['entities'] else ''
            words = list(row['words']) if row['words'] is not None else []
            cat = categorize_receipt(company, words)
            label = CLASS_TO_IDX[cat]
            
            img_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            tensor = resize_transform(img)
            
            tensors.append(tensor)
            labels.append(label)
            keys.append(key)
            
            records_meta.append({
                "idx": idx,
                "key": key,
                "category": cat,
                "label": label,
                "orig_size": [row['image_size']['width'], row['image_size']['height']] if row['image_size'] else [img.width, img.height],
                "num_words": len(words),
                "entities": {
                    "company": row['entities'].get('company', '') if row['entities'] else '',
                    "date": row['entities'].get('date', '') if row['entities'] else '',
                    "address": row['entities'].get('address', '') if row['entities'] else '',
                    "total": row['entities'].get('total', '') if row['entities'] else '',
                }
            })
            
            if (idx + 1) % 100 == 0 or idx == len(df) - 1:
                print(f"Processed {idx + 1}/{len(df)} images ({time.time() - start_t:.1f}s)")
                
        stacked_tensors = torch.stack(tensors) # shape: (N, 3, 224, 224)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        cache_data = {
            "images": stacked_tensors,
            "labels": labels_tensor,
            "keys": keys,
            "metadata": records_meta,
            "classes": CLASS_NAMES
        }
        
        torch.save(cache_data, cache_file)
        meta_json_file = f"artifacts/cache/{split_name}_metadata.json"
        with open(meta_json_file, "w") as f:
            json.dump(records_meta, f, indent=2)
            
        file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"Saved {split_name} cache: {cache_file} ({file_size_mb:.2f} MB) in {time.time() - start_t:.2f}s")

if __name__ == "__main__":
    build_cache()
