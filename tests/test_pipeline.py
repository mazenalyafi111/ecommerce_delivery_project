# -*- coding: utf-8 -*-
import os
import unittest
import pandas as pd
import numpy as np

PROCESSED_DATA_PATH = os.path.join("data", "processed")

class TestEcommercePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_path = os.path.join(PROCESSED_DATA_PATH, "orders_features_modeling.csv")
        assert os.path.exists(cls.file_path), "ملف النمذجة غير موجود! يرجى تشغيل feature_engineering.py أولاً."
        cls.df = pd.read_csv(cls.file_path)

    def test_unique_order_keys(self):
        """اختبار عدم تضاعف الطلبات وتفرد المفتاح الأساسي order_id"""
        self.assertEqual(
            self.df["order_id"].nunique(),
            len(self.df),
            "خطأ: يوجد تكرار في معرفات الطلبات order_id!"
        )

    def test_no_forbidden_leakage_columns(self):
        """اختبار استبعاد الحقول المحظورة ومنع تسرب البيانات"""
        forbidden_cols = [
            "order_status",
            "order_delivered_customer_date",
            "order_delivered_carrier_date"
        ]
        for col in forbidden_cols:
            self.assertNotIn(
                col,
                self.df.columns,
                "خطأ تسرب بيانات: العمود المحظور {} موجود في جدول النمذجة!".format(col)
            )

    def test_target_variable_logic(self):
        """اختبار صحة تعريف الهدف وحالاته الحدية"""
        # التأكد من أن الهدف ثنائي القيم فقط (0 أو 1)
        unique_targets = set(self.df["is_late"].dropna().unique())
        self.assertTrue(unique_targets.issubset({0, 1}), "خطأ: المتغير المستهدف يحتوي قيماً خارج {0, 1}!")
        self.assertEqual(self.df["is_late"].isnull().sum(), 0, "خطأ: توجد قيم مفقودة في المتغير المستهدف!")

    def test_engineered_features_validity(self):
        """اختبار سلامة الخصائص المشتقة (مثل عدم وجود أيام سالبة للتسليم المخطط)"""
        self.assertTrue(
            (self.df["planned_delivery_days"] >= 0).all(),
            "خطأ: توجد قيم سالبة في المدة المخططة للتسليم!"
        )
        self.assertTrue(
            self.df["is_same_state"].isin([0, 1]).all(),
            "خطأ: خاصية is_same_state يجب أن تكون 0 أو 1 حصراً!"
        )
        self.assertTrue(
            self.df["has_multiple_sellers"].isin([0, 1]).all(),
            "خطأ: خاصية has_multiple_sellers يجب أن تكون 0 أو 1 حصراً!"
        )

if __name__ == "__main__":
    unittest.main()