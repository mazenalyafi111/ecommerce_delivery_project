# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report
)

PROCESSED_DATA_PATH = os.path.join("data", "processed")
REPORTS_PATH = "reports"
os.makedirs(REPORTS_PATH, exist_ok=True)

def train_and_evaluate_models():
    print("Loading modeling dataset...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "orders_features_modeling.csv"),
        parse_dates=["order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"]
    )
    
    # 1. الترتيب والتقسيم الزمني (Train / Validation / Test)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print("Temporal Split:")
    print(" - Train set     : {:,} orders (Up to {})".format(len(train_df), train_df['order_purchase_timestamp'].max().strftime('%Y-%m-%d')))
    print(" - Validation set: {:,} orders (Up to {})".format(len(val_df), val_df['order_purchase_timestamp'].max().strftime('%Y-%m-%d')))
    print(" - Test set      : {:,} orders (Up to {})".format(len(test_df), test_df['order_purchase_timestamp'].max().strftime('%Y-%m-%d')))

    # تحديد الخصائص المدخلة والهدف (بدون معرفات أو حقول تسريب)
    numeric_features = [
        "planned_delivery_days", "approval_delay_hours", "purchase_hour",
        "num_items", "num_sellers", "total_price", "total_freight",
        "total_weight_g", "total_volume_cm3", "freight_ratio", "freight_per_item"
    ]
    categorical_features = [
        "is_same_state", "has_multiple_sellers", "purchase_dayofweek",
        "customer_state", "primary_seller_state"
    ]
    target = "is_late"

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]

    X_val = val_df[numeric_features + categorical_features]
    y_val = val_df[target]

    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    # خطوط المعالجة المسبقة لمنع التسرب
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # تعريف النماذج الثلاثة المطلوبة
    models = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, max_depth=8, class_weight="balanced", random_state=42
        )
    }

    results = []

    print("\n" + "=" * 70)
    print("Evaluating models on Validation and Test Sets...")
    print("=" * 70)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        
        # التدريب على بيانات التدريب فقط
        pipe.fit(X_train, y_train)

        # التقييم على بيانات الاختبار النهائي (Hold-out Test Set)
        y_pred = pipe.predict(X_test)
        
        # حساب الاحتمال لمتوسط الإحكام
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            ap = average_precision_score(y_test, y_proba)
        else:
            ap = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Average Precision (PR-AUC)": round(ap, 4) if not np.isnan(ap) else "N/A"
        })

        print(f"\n--- {name} ---")
        print(f"Confusion Matrix:\n{cm}")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]} | FN: {cm[1,0]}, TP: {cm[1,1]}")
        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Model Evaluation Summary (Test Set):")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(REPORTS_PATH, "model_comparison_results.csv"), index=False)
    print("\nSaved evaluation summary table to: reports/model_comparison_results.csv")

if __name__ == "__main__":
    train_and_evaluate_models()