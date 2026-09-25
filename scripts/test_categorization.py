import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import re

def test_categorization():
    train_df = pq.read_table("train-00000-of-00001.parquet").to_pandas()
    test_df = pq.read_table("test-00000-of-00001.parquet").to_pandas()

    def categorize_receipt(company_name, words):
        c = (company_name or '').upper()
        all_words_text = " ".join([str(w).upper() for w in (words if words is not None else [])])
        full_text = f"{c} {all_words_text}"

        # 1. Hardware & Home Improvement
        if any(k in full_text for k in ["MR. D.I.Y.", "MR D.I.Y", "MR DIY", "HARDWARE", "PAPAN", "TIMBER", "BUILDING", "HOME DECO", "AIK HUAT", "ELECTRICAL"]):
            return "Hardware & Home"
        
        # 2. Bakery & Confectionery
        if any(k in full_text for k in ["GARDENIA", "BAKERY", "BAKERIES", "CONFECTIONERY", "CAKE", "BREAD", "ROTI", "PASTRY"]):
            return "Bakery & Confectionery"
        
        # 3. Supermarket & Grocery
        if any(k in full_text for k in ["SPEED MART", "SPEEDMART", "99 SPEED", "AEON", "GROCER", "SUPERMARKET", "PASARAYA", "HYPERMARKET", "MINI MARKET", "MART"]):
            return "Supermarket & Grocery"
        
        # 4. Restaurant & Food Service
        if any(k in full_text for k in ["RESTORAN", "RESTAURANT", "FOOD", "KOPITIAM", "CAFE", "COFFEE", "SEAFOOD", "UNIHAKKA", "GERBANG ALAF", "STEAMBOAT", "KITCHEN", "DINING", "MCDONALD"]):
            return "Restaurant & Dining"
            
        # 5. Stationery & Bookstore
        if any(k in full_text for k in ["STATIONERY", "BOOK", "POPULAR BOOK", "PAPER", "PEN", "OFFICE SUPPLIES", "PRINTING"]):
            return "Stationery & Bookstore"
            
        # 6. Pharmacy & Health/Beauty
        if any(k in full_text for k in ["PHARMACY", "HEALTH", "BEAUTY", "GUARDIAN", "WATSONS", "FARMASI", "CLINIC", "MEDICAL"]):
            return "Pharmacy & Health"
            
        return "General Retail & Services"

    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        cats = [categorize_receipt(row['entities'].get('company'), row['words']) for idx, row in df.iterrows()]
        s = pd.Series(cats)
        print(f"\n--- {name} Category Distribution (Total {len(df)}) ---")
        print(s.value_counts())
        print(s.value_counts(normalize=True) * 100)

if __name__ == "__main__":
    test_categorization()
