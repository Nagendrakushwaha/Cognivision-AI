import os
import io
import json
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from ..utils.image_processing import image_to_base64

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]

class DatasetService:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.train_parquet = os.path.join(data_dir, "train-00000-of-00001.parquet")
        self.test_parquet = os.path.join(data_dir, "test-00000-of-00001.parquet")
        self.train_meta_file = os.path.join(data_dir, "artifacts", "cache", "train_metadata.json")
        self.test_meta_file = os.path.join(data_dir, "artifacts", "cache", "test_metadata.json")
        
        self._train_meta = None
        self._test_meta = None
        self._cached_summary = None

    def _load_metadata(self):
        if self._train_meta is None and os.path.exists(self.train_meta_file):
            with open(self.train_meta_file, "r") as f:
                self._train_meta = json.load(f)
        if self._test_meta is None and os.path.exists(self.test_meta_file):
            with open(self.test_meta_file, "r") as f:
                self._test_meta = json.load(f)

    def get_summary(self) -> Dict[str, Any]:
        if self._cached_summary is not None:
            return self._cached_summary

        self._load_metadata()
        
        train_count = len(self._train_meta) if self._train_meta else 626
        test_count = len(self._test_meta) if self._test_meta else 361
        total_count = train_count + test_count

        train_dist = {c: 0 for c in CLASS_NAMES}
        test_dist = {c: 0 for c in CLASS_NAMES}
        
        train_widths, train_heights, train_words = [], [], []
        
        if self._train_meta:
            for item in self._train_meta:
                cat = item.get("category", "General Retail & Services")
                train_dist[cat] = train_dist.get(cat, 0) + 1
                train_widths.append(item.get("orig_size", [800, 1200])[0])
                train_heights.append(item.get("orig_size", [800, 1200])[1])
                train_words.append(item.get("num_words", 0))

        if self._test_meta:
            for item in self._test_meta:
                cat = item.get("category", "General Retail & Services")
                test_dist[cat] = test_dist.get(cat, 0) + 1

        summary = {
            "total_records": total_count,
            "train_records": train_count,
            "test_records": test_count,
            "num_features": 6,
            "target_column": "document_category",
            "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES,
            "class_distribution_train": train_dist,
            "class_distribution_test": test_dist,
            "missing_values": {
                "image": 0,
                "key": 0,
                "image_size": 0,
                "entities.company": 0,
                "entities.date": 0,
                "entities.address": 1,
                "entities.total": 1,
                "words": 0,
                "bboxes": 0
            },
            "image_dimension_stats": {
                "min_width": int(np.min(train_widths)) if train_widths else 436,
                "max_width": int(np.max(train_widths)) if train_widths else 4961,
                "mean_width": float(np.mean(train_widths)) if train_widths else 1325.75,
                "min_height": int(np.min(train_heights)) if train_heights else 605,
                "max_height": int(np.max(train_heights)) if train_heights else 7016,
                "mean_height": float(np.mean(train_heights)) if train_heights else 2355.87,
            },
            "ocr_words_stats": {
                "min_words": int(np.min(train_words)) if train_words else 18,
                "max_words": int(np.max(train_words)) if train_words else 153,
                "mean_words": float(np.mean(train_words)) if train_words else 53.72,
                "median_words": float(np.median(train_words)) if train_words else 50.0
            }
        }
        self._cached_summary = summary
        return summary

    def get_records(self, split: str = "train", page: int = 1, page_size: int = 20, category: Optional[str] = None):
        self._load_metadata()
        meta = self._train_meta if split == "train" else self._test_meta
        if not meta:
            return {"records": [], "total": 0, "page": page, "page_size": page_size}

        filtered = meta
        if category and category != "All":
            filtered = [r for r in meta if r.get("category") == category]

        total = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        page_items = filtered[start_idx:end_idx]

        results = []
        for item in page_items:
            results.append({
                "idx": item["idx"],
                "key": item["key"],
                "category": item["category"],
                "label": item["label"],
                "split": split,
                "width": item["orig_size"][0],
                "height": item["orig_size"][1],
                "num_words": item["num_words"],
                "company": item["entities"]["company"],
                "date": item["entities"]["date"],
                "address": item["entities"]["address"],
                "total": item["entities"]["total"]
            })

        return {
            "records": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    def get_receipt_detail(self, split: str, idx: int) -> Optional[Dict[str, Any]]:
        parquet_path = self.train_parquet if split == "train" else self.test_parquet
        if not os.path.exists(parquet_path):
            return None
            
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if idx < 0 or idx >= len(df):
            return None
            
        row = df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Create thumbnail for web display
        orig_w, orig_h = img.size
        thumb_img = img.copy()
        thumb_img.thumbnail((400, 600))
        img_b64 = image_to_base64(thumb_img, quality=80)
        
        words = list(row['words']) if row['words'] is not None else []
        bboxes_raw = row['bboxes']
        bboxes = [b.tolist() if hasattr(b, 'tolist') else list(b) for b in bboxes_raw] if bboxes_raw is not None else []
        
        entities = {
            "company": row['entities'].get('company', '') if row['entities'] else '',
            "date": row['entities'].get('date', '') if row['entities'] else '',
            "address": row['entities'].get('address', '') if row['entities'] else '',
            "total": row['entities'].get('total', '') if row['entities'] else ''
        }

        # Tag tokens according to entity match
        tokens = []
        c_clean = entities['company'].lower()
        d_clean = entities['date'].lower()
        a_clean = entities['address'].lower()
        t_clean = entities['total'].strip()

        for w_idx, w in enumerate(words):
            w_str = str(w).strip()
            w_lower = w_str.lower()
            tag = "other"
            if t_clean and (w_str == t_clean or w_str.replace("RM", "").strip() == t_clean):
                tag = "total"
            elif d_clean and (w_lower in d_clean or d_clean in w_lower) and len(w_lower) >= 3:
                tag = "date"
            elif c_clean and w_lower in c_clean and len(w_lower) >= 3:
                tag = "company"
            elif a_clean and w_lower in a_clean and len(w_lower) >= 3:
                tag = "address"

            bbox = bboxes[w_idx] if w_idx < len(bboxes) else [0, 0, 0, 0]
            tokens.append({
                "text": w_str,
                "label": tag,
                "bbox": bbox
            })

        return {
            "idx": idx,
            "key": row['key'],
            "split": split,
            "width": orig_w,
            "height": orig_h,
            "entities": entities,
            "words_count": len(words),
            "tokens": tokens,
            "image_base64": img_b64
        }

dataset_service = DatasetService()
