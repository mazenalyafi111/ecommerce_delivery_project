# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DATA_PATH = os.path.join("data", "processed")
FIGURES_PATH = "figures"
REPORTS_PATH = "reports"

os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# تحسين مظهر الرسوم البيانية
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def run_eda_and_generate_figures():
    print("Loading data for EDA...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "orders_features_modeling.csv"),
        parse_dates=["order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"]
    )

    # -------------------------------------------------------------
    # 0. إثبات استخدام 3 عمليات عددية عبر NumPy (متطلب التقييم)
    # -------------------------------------------------------------
    # عملية 1: حساب متجهي للانحراف بين المدة المخططة ومتوسطها
    mean_planned = np.mean(df["planned_delivery_days"])
    df["planned_days_deviation"] = np.subtract(df["planned_delivery_days"], mean_planned)

    # عملية 2: تطبيق تحويل لوغاريتمي على إجمالي السعر والشحن للتعامل مع الالتواء الشديد
    df["log_total_price"] = np.log1p(df["total_price"].clip(lower=0))
    df["log_total_freight"] = np.log1p(df["total_freight"].clip(lower=0))

    # عملية 3: قناع منطقي لفرز الطلبات ذات الشحن المرتفع
    high_freight_threshold = np.percentile(df["total_freight"], 90)
    high_freight_mask = np.greater(df["total_freight"].values, high_freight_threshold)
    df["is_high_freight"] = high_freight_mask.astype(int)

    print("NumPy operations applied successfully.")

    # -------------------------------------------------------------
    # إجابة الأسئلة التحليلية الستة وحفظ جداولها وتوليد الرسوم الـ 8
    # -------------------------------------------------------------

    # === السؤال 1 والرسم 1: أعداد الطلبات ونسبة التأخر عبر الزمن ===
    df["purchase_year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    # استبعاد الأشهر الحدية ذات السجلات القليلة جداً
    monthly = df.groupby("purchase_year_month").agg(
        order_count=("order_id", "count"),
        late_rate=("is_late", "mean")
    ).reset_index()
    monthly = monthly[(monthly["order_count"] > 50) & (monthly["purchase_year_month"] < "2018-09")]
    monthly["late_percentage"] = monthly["late_rate"] * 100

    # الرسم 1 (Line Plot / Dual Axis)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Year-Month')
    ax1.set_ylabel('Total Orders', color=color)
    ax1.plot(monthly['purchase_year_month'], monthly['order_count'], marker='o', color=color, linewidth=2, label="Orders Volume")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Late Delivery Rate (%)', color=color)
    ax2.plot(monthly['purchase_year_month'], monthly['late_percentage'], marker='s', color=color, linestyle='--', linewidth=2, label="Late %")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Figure 1: Monthly Order Volume vs Late Delivery Rate (Temporal Dynamics)', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig1_temporal_order_late_trend.png"), dpi=300)
    plt.close()

    # === السؤال 2 والرسم 2 + 3: المسارات بين الولايات وتعدد البائعين ===
    # مقارنة داخل نفس الولاية مقابل عبر الولايات
    state_perf = df.groupby("is_same_state").agg(
        sample_size=("order_id", "count"),
        late_rate=("is_late", "mean")
    ).reset_index()
    state_perf["route_type"] = state_perf["is_same_state"].map({1: "Intra-State (Same)", 0: "Inter-State (Different)"})
    state_perf["late_percentage"] = state_perf["late_rate"] * 100

    # الرسم 2 (Bar Chart مع حجم العينة)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(state_perf["route_type"], state_perf["late_percentage"], color=['#2ca02c', '#d62728'], width=0.5)
    plt.ylabel("Late Delivery Rate (%)")
    plt.title("Figure 2: Late Rate by Shipping Route Type (Intra vs Inter-State)")
    for bar, size in zip(bars, state_perf["sample_size"]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}%\n(n={size:,})", ha='center', va='bottom', fontsize=9)
    plt.ylim(0, state_perf["late_percentage"].max() + 3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig2_route_type_late_rate.png"), dpi=300)
    plt.close()

    # أثر تعدد البائعين
    multi_seller = df.groupby("has_multiple_sellers").agg(
        sample_size=("order_id", "count"),
        late_rate=("is_late", "mean")
    ).reset_index()
    multi_seller["seller_type"] = multi_seller["has_multiple_sellers"].map({0: "Single Seller", 1: "Multiple Sellers"})
    multi_seller["late_percentage"] = multi_seller["late_rate"] * 100

    # الرسم 3 (Bar Chart لتعدد البائعين)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(multi_seller["seller_type"], multi_seller["late_percentage"], color=['#1f77b4', '#ff7f0e'], width=0.5)
    plt.ylabel("Late Delivery Rate (%)")
    plt.title("Figure 3: Delay Impact of Multiple Sellers per Order")
    for bar, size in zip(bars, multi_seller["sample_size"]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%\n(n={size:,})", ha='center', va='bottom', fontsize=9)
    plt.ylim(0, multi_seller["late_percentage"].max() + 3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig3_multiple_sellers_late_rate.png"), dpi=300)
    plt.close()

    # === السؤال 3 والرسم 4 + 5: فئات المنتجات وحجم الطلب ===
    # أعلى 12 فئة منتجات من حيث عدد الطلبات
    top_categories = df["primary_product_category"].value_counts().head(12).index
    cat_df = df[df["primary_product_category"].isin(top_categories)].groupby("primary_product_category").agg(
        sample_size=("order_id", "count"),
        late_rate=("is_late", "mean")
    ).reset_index()
    cat_df["late_percentage"] = cat_df["late_rate"] * 100
    cat_df = cat_df.sort_values(by="late_percentage", ascending=False)

    # الرسم 4 (Horizontal Bar Chart)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(cat_df["primary_product_category"], cat_df["late_percentage"], color="#3470a3")
    plt.xlabel("Late Delivery Rate (%)")
    plt.title("Figure 4: Late Rate Across Top Product Categories (with Sample Sizes)")
    for bar, size in zip(bars, cat_df["sample_size"]):
        xval = bar.get_width()
        plt.text(xval + 0.2, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}% (n={size:,})", va='center', fontsize=8)
    plt.xlim(0, cat_df["late_percentage"].max() + 3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig4_categories_late_rate.png"), dpi=300)
    plt.close()

    # أثر حجم الطلب (عدد القطع) عبر Boxplot للمقارنة
    # الرسم 5 (Boxplot)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="is_late", y="total_weight_g", data=df[df["total_weight_g"] < 15000], palette=["#2ca02c", "#d62728"])
    plt.xticks([0, 1], ["On-Time Orders (0)", "Late Orders (1)"])
    plt.ylabel("Total Weight (g) [Filtered < 15kg for clarity]")
    plt.xlabel("Delivery Status")
    plt.title("Figure 5: Product Weight Distribution by Delivery Outcome (Boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig5_box_weight_vs_late.png"), dpi=300)
    plt.close()

    # === السؤال 4 والرسم 6: المدة المخططة للتسليم والتأخر الفعلي ===
    # الرسم 6 (Histogram / KDE)
    plt.figure(figsize=(9, 5))
    sns.histplot(df[df["is_late"] == 0]["planned_delivery_days"], color="blue", label="On-Time", kde=True, stat="density", bins=30, alpha=0.4)
    sns.histplot(df[df["is_late"] == 1]["planned_delivery_days"], color="red", label="Late", kde=True, stat="density", bins=30, alpha=0.4)
    plt.xlim(0, 50)
    plt.xlabel("Planned Delivery Duration (Days)")
    plt.ylabel("Density")
    plt.title("Figure 6: Density Distribution of Planned Delivery Days by Delivery Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig6_planned_days_density.png"), dpi=300)
    plt.close()

    # === السؤال 5 والرسم 7: توزيع الفئة المستهدفة ومناقشة عدم التوازن ===
    # الرسم 7 (Target Distribution Bar)
    target_counts = df["is_late"].value_counts().reset_index()
    target_counts.columns = ["is_late", "count"]
    target_counts["label"] = target_counts["is_late"].map({0: "On-Time (93.23%)", 1: "Late (6.77%)"})

    plt.figure(figsize=(7, 5))
    bars = plt.bar(target_counts["label"], target_counts["count"], color=["#1f77b4", "#d62728"], width=0.4)
    plt.ylabel("Number of Orders")
    plt.title("Figure 7: Severe Class Imbalance in Target Variable (is_late)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f"{yval:,}", ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, 105000)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig7_target_imbalance.png"), dpi=300)
    plt.close()

    # === السؤال 6 والرسم 8: مصفوفة العلاقات والارتباطات (Heatmap) ===
    corr_cols = [
        "is_late", "planned_delivery_days", "is_same_state", "approval_delay_hours",
        "num_items", "total_price", "total_freight", "total_weight_g", "has_multiple_sellers"
    ]
    corr_matrix = df[corr_cols].corr()

    # الرسم 8 (Correlation Heatmap)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Figure 8: Correlation Matrix of Key Order Features with Target", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "fig8_correlation_heatmap.png"), dpi=300)
    plt.close()

    print("All 8 figures generated and saved to 'figures/' folder.")

if __name__ == "__main__":
    run_eda_and_generate_figures()