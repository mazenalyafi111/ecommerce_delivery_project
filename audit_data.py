# -*- coding: utf-8 -*-
import os
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw")

DATASET_FILES = {
    "Orders": "olist_orders_dataset.csv",
    "Customers": "olist_customers_dataset.csv",
    "Order Items": "olist_order_items_dataset.csv",
    "Payments": "olist_order_payments_dataset.csv",
    "Products": "olist_products_dataset.csv",
    "Sellers": "olist_sellers_dataset.csv",
    "Category Translation": "product_category_name_translation.csv"
}

def inspect_datasets():
    summary_records = []
    
    print("=" * 70)
    print("Starting initial audit of Olist datasets...")
    print("=" * 70)

    for table_name, file_name in DATASET_FILES.items():
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        
        if not os.path.exists(file_path):
            print("[Warning] File not found: " + file_name + " in " + RAW_DATA_PATH)
            continue
            
        df = pd.read_csv(file_path)
        
        num_rows, num_cols = df.shape
        missing_cells = df.isnull().sum().sum()
        cols_with_missing = (df.isnull().sum() > 0).sum()
        
        summary_records.append({
            "Table": table_name,
            "File Name": file_name,
            "Rows": "{:,}".format(num_rows),
            "Columns": num_cols,
            "Missing Values": "{:,}".format(missing_cells),
            "Cols with Missing": cols_with_missing
        })
        
        print("\nChecked [{0}]: {1:,} rows x {2} cols.".format(table_name, num_rows, num_cols))
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            print("   Columns containing missing values:")
            for col, count in cols_with_nulls.items():
                pct = (float(count) / num_rows) * 100
                print("     - {0}: {1:,} ({2:.2f}%)".format(col, count, pct))

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        print("\n" + "=" * 70)
        print("Initial Data Audit Summary (For Project Report):")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        
        if not os.path.exists("reports"):
            os.makedirs("reports")
        summary_df.to_csv(os.path.join("reports", "initial_data_audit.csv"), index=False, encoding="utf-8-sig")
        print("\nSummary table saved to: reports/initial_data_audit.csv")

if __name__ == "__main__":
    inspect_datasets()