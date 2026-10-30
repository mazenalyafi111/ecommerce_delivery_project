# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def load_raw_data():
    """قراءة الجداول الأساسية مع تحديد بادئات الرموز البريدية كنصوص."""
    orders = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_orders_dataset.csv"),
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )
    
    customers = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_customers_dataset.csv"),
        dtype={"customer_zip_code_prefix": str}
    )
    
    order_items = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_order_payments_dataset.csv"))
    products = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_products_dataset.csv"))
    
    sellers = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_sellers_dataset.csv"),
        dtype={"seller_zip_code_prefix": str}
    )
    
    translations = pd.read_csv(os.path.join(RAW_DATA_PATH, "product_category_name_translation.csv"))
    
    return orders, customers, order_items, payments, products, sellers, translations

def aggregate_order_items(order_items, products, sellers, translations):
    """
    دمج تفاصيل العناصر مع المنتجات والبائعين أولاً،
    ثم تجميعها إلى مستوى الطلب (صف واحد لكل order_id).
    """
    # ربط المنتجات بترجمتها الإنجليزية
    products_trans = products.merge(translations, on="product_category_name", how="left")
    products_trans["product_category_name_english"] = products_trans["product_category_name_english"].fillna("other")
    
    # دمج العناصر مع خصائص المنتجات والبائعين
    items_enriched = order_items.merge(
        products_trans[[
            "product_id", "product_category_name_english",
            "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
        ]],
        on="product_id",
        how="left"
    )
    items_enriched = items_enriched.merge(
        sellers[["seller_id", "seller_state", "seller_city"]],
        on="seller_id",
        how="left"
    )

    # حساب حجم المنتج بالسنتيمتر المكعب (عملية عددية بمكتبة NumPy/Pandas)
    items_enriched["product_volume_cm3"] = (
        items_enriched["product_length_cm"] * 
        items_enriched["product_height_cm"] * 
        items_enriched["product_width_cm"]
    )

    # التجميع لمستوى الطلب
    grouped = items_enriched.groupby("order_id").agg(
        num_items=("order_item_id", "count"),
        num_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("product_volume_cm3", "sum"),
        primary_seller_state=("seller_state", "first"),
        primary_product_category=("product_category_name_english", "first")
    ).reset_index()

    return grouped

def aggregate_payments(payments):
    """
    تجميع سجلات الدفع المتعددة لكل طلب إلى مستوى الطلب (order_id).
    """
    grouped = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        num_payment_installments=("payment_installments", "max"),
        num_payment_types=("payment_type", "nunique"),
        primary_payment_type=("payment_type", "first")
    ).reset_index()
    
    return grouped

def build_master_dataset():
    """بناء الجدول الشامل المدمج بعد التحقق من تفرد الطلبات."""
    print("Loading raw files...")
    orders, customers, order_items, payments, products, sellers, translations = load_raw_data()
    
    print("Aggregating order items...")
    agg_items = aggregate_order_items(order_items, products, sellers, translations)
    
    print("Aggregating payments...")
    agg_payments = aggregate_payments(payments)
    
    print("Merging master table...")
    # ربط الطلبات مع العملاء
    master = orders.merge(
        customers[["customer_id", "customer_unique_id", "customer_state", "customer_city"]],
        on="customer_id",
        how="left"
    )
    
    # ربط العناصر المجمعة
    master = master.merge(agg_items, on="order_id", how="left")
    
    # ربط المدفوعات المجمعة
    master = master.merge(agg_payments, on="order_id", how="left")
    
    # التحقق من أن عدد الصفوف لم يتضاعف
    assert len(master) == len(orders), "خطأ: حدث تضاعف في أعداد الطلبات بعد الدمج!"
    print("Master dataset verified: {} orders (1 row per order)".format(len(master)))
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_file = os.path.join(PROCESSED_DATA_PATH, "orders_master.csv")
    master.to_csv(out_file, index=False)
    print("Master file saved successfully to: {}".format(out_file))

if __name__ == "__main__":
    build_master_dataset()