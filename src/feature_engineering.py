# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

PROCESSED_DATA_PATH = os.path.join("data", "processed")

def engineer_features_and_target():
    print("Loading master dataset...")
    master_path = os.path.join(PROCESSED_DATA_PATH, "orders_master.csv")
    df = pd.read_csv(
        master_path,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )
    
    initial_count = len(df)
    print("Initial orders count: {:,}".format(initial_count))
    
    # 1. تصفية مجتمع الدراسة (Eligible Cohort) بناءً على شروط المشروع الصريحة:
    # - يجب أن يكون الطلب مسلماً مع تواريخ صالحة للاعتماد والتسليم والتوقع
    valid_mask = (
        df["order_status"] == "delivered"
    ) & df["order_approved_at"].notna() & df["order_delivered_customer_date"].notna() & df["order_estimated_delivery_date"].notna()
    
    df_clean = df[valid_mask].copy()
    excluded_count = initial_count - len(df_clean)
    print("Excluded invalid/undelivered orders: {:,}".format(excluded_count))
    print("Eligible modeling orders: {:,}".format(len(df_clean)))
    
    # 2. تعريف المتغير المستهدف is_late:
    # مقارنة أجزاء التاريخ (Day level) حصراً وليس أوقات منتصف الليل
    actual_delivery_date = df_clean["order_delivered_customer_date"].dt.date
    estimated_delivery_date = df_clean["order_estimated_delivery_date"].dt.date
    
    # Target = 1 إذا وصل في يوم لاحق للمتوقع، و0 خلاف ذلك
    df_clean["is_late"] = (actual_delivery_date > estimated_delivery_date).astype(int)
    
    late_rate = df_clean["is_late"].mean() * 100
    print("Late delivery rate: {:.2f}% (Total late: {:,})".format(late_rate, df_clean["is_late"].sum()))
    
    # 3. اشتقاق الخصائص الثمانية المسموحة عند لحظة الاعتماد:
    # خاصية 1: المدة المخططة للتسليم بالأيام
    df_clean["planned_delivery_days"] = (
        (df_clean["order_estimated_delivery_date"] - df_clean["order_approved_at"]).dt.total_seconds() / (24 * 3600)
    ).clip(lower=0)
    
    # خاصية 2: تطابق ولاية البائع والعميل
    df_clean["is_same_state"] = (df_clean["customer_state"] == df_clean["primary_seller_state"]).astype(int)
    
    # خاصية 3: التأخر في الاعتماد بالساعات
    df_clean["approval_delay_hours"] = (
        (df_clean["order_approved_at"] - df_clean["order_purchase_timestamp"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    
    # خاصية 4: يوم الشراء (0 إلى 6)
    df_clean["purchase_dayofweek"] = df_clean["order_purchase_timestamp"].dt.dayofweek
    
    # خاصية 5: ساعة الشراء (0 إلى 23)
    df_clean["purchase_hour"] = df_clean["order_purchase_timestamp"].dt.hour
    
    # خاصية 6: نسبة الشحن إلى إجمالي السعر
    df_clean["freight_ratio"] = df_clean["total_freight"] / (df_clean["total_price"] + 1e-5)
    
    # خاصية 7: تكلفة الشحن لكل قطعة
    df_clean["freight_per_item"] = df_clean["total_freight"] / df_clean["num_items"].clip(lower=1)
    
    # خاصية 8: تعدد البائعين
    df_clean["has_multiple_sellers"] = (df_clean["num_sellers"] > 1).astype(int)
    
    # 4. منع تسرب البيانات (Data Leakage Exclusion):
    # عزل واستبعاد حقول ما بعد لحظة الاعتماد من جدول النمذجة
    leakage_cols = [
        "order_status",
        "order_delivered_carrier_date",
        "order_delivered_customer_date"
    ]
    df_modeling = df_clean.drop(columns=leakage_cols)
    
    # حفظ الجدول النهائي الجاهز للتحليل والنمذجة
    out_path = os.path.join(PROCESSED_DATA_PATH, "orders_features_modeling.csv")
    df_modeling.to_csv(out_path, index=False)
    print("Modeling dataset saved successfully to: {}".format(out_path))

if __name__ == "__main__":
    engineer_features_and_target()