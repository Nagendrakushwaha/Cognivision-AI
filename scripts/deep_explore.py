import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import re

def explore():
    train_table = pq.read_table("train-00000-of-00001.parquet")
    df = train_table.to_pandas()
    
    print("Train records count:", len(df))
    
    # Check sample row 0 words and how they align with entities
    row0 = df.iloc[0]
    print("\nRow 0 entities:", row0['entities'])
    print("Row 0 words count:", len(row0['words']))
    print("Row 0 words:", row0['words'][:15])
    
    # Let's see how token labeling works
    # For each word in words, check which entity it belongs to
    def tag_words(words, entities):
        tags = []
        company_clean = (entities.get('company') or '').lower()
        date_clean = (entities.get('date') or '').lower()
        address_clean = (entities.get('address') or '').lower()
        total_clean = (entities.get('total') or '').strip()
        
        for w in words:
            w_str = str(w).strip()
            w_lower = w_str.lower()
            if not w_str:
                tags.append('OTHER')
            elif total_clean and (w_str == total_clean or re.sub(r'[^\d.]', '', w_str) == total_clean):
                tags.append('TOTAL')
            elif date_clean and (w_lower in date_clean or date_clean in w_lower) and len(w_lower) >= 3:
                tags.append('DATE')
            elif company_clean and w_lower in company_clean and len(w_lower) >= 3:
                tags.append('COMPANY')
            elif address_clean and w_lower in address_clean and len(w_lower) >= 3:
                tags.append('ADDRESS')
            else:
                tags.append('OTHER')
        return tags

    all_tags = []
    for idx, row in df.iterrows():
        tags = tag_words(row['words'], row['entities'])
        all_tags.extend(tags)
        
    tag_counts = pd.Series(all_tags).value_counts()
    print("\n--- Token Entity Label Distribution across Train (Total Tokens: {}) ---".format(len(all_tags)))
    print(tag_counts)
    print(tag_counts / len(all_tags) * 100)

    # Let's also check receipt classification / categories
    # What merchant industries exist in SROIE?
    # e.g., Bakeries, Supermarkets/Retail, Restaurants/Dining, Stationery/Books, Home/Hardware, Services
    # Let's inspect unique companies
    companies = [row['entities'].get('company', '') for idx, row in df.iterrows() if row['entities']]
    comp_series = pd.Series(companies)
    print("\nTotal unique companies:", comp_series.nunique())
    print("Top 15 companies:")
    print(comp_series.value_counts().head(15))

if __name__ == "__main__":
    explore()
