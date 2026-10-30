# -*- coding: utf-8 -*-
import subprocess
import sys

def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f">> {desc}...")
    print("=" * 70)
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"Error occurred while executing: {desc}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("Executing Full End-to-End Pipeline for DPD601 Project")
    
    # 1. تدقيق وفحص البيانات الخام
    run_command(["audit_data.py"], "Step 1: Raw Data Audit")
    
    # 2. تجميع البيانات ودمجها لمنع التكرار
    run_command(["src/preprocessing.py"], "Step 2: Aggregation and Merging")
    
    # 3. هندسة الخصائص ومنع التسرب
    run_command(["src/feature_engineering.py"], "Step 3: Feature Engineering and Target Formulation")
    
    # 4. تشغيل الاختبارات الآلية
    run_command(["-m", "unittest", "tests/test_pipeline.py"], "Step 4: Automated Testing")
    
    # 5. التحليل الاستكشافي والرسوم البيانية الثمانية
    run_command(["src/eda_visuals.py"], "Step 5: Exploratory Data Analysis & Visualizations")
    
    # 6. تدريب وتقييم النماذج والمقارنة الزمنية
    run_command(["src/models.py"], "Step 6: Machine Learning Training and Evaluation")
    
    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("Figures saved to: figures/")
    print("Reports & Summaries saved to: reports/")
    print("=" * 70)