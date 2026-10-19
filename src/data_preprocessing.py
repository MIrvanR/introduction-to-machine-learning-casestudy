"""
data_preprocessing.py
Tahap 1 & 2 dalam Siklus Machine Learning:
- Pengumpulan & Persiapan Data (Data Preparation & Cleaning)
- Pembagian Data menjadi Data Latih (Training) dan Data Uji (Testing)

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class MLDataPipeline:
    def __init__(self, data_path="data/heart_disease_risk.csv", test_size=0.20, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.raw_df = None
        self.clean_df = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        
        # Fitur prediktor
        self.feature_columns = [
            "age", "sex", "chest_pain_type", "resting_bp",
            "cholesterol", "fasting_blood_sugar", "max_heart_rate",
            "exercise_angina", "oldpeak"
        ]
        
    def load_data(self):
        """Memuat dataset mentah dari file CSV."""
        self.raw_df = pd.read_csv(self.data_path)
        print(f"[TAHAP 1] Data berhasil dimuat: {self.raw_df.shape[0]} baris, {self.raw_df.shape[1]} kolom")
        return self.raw_df

    def inspect_and_clean(self):
        """
        Pembersihan data:
        1. Pemeriksaan missing values
        2. Imputasi nilai hilang dengan median (mencegah distorsi outlier)
        3. Pemeriksaan distribusi kelas target
        """
        if self.raw_df is None:
            self.load_data()
            
        print("\n--- Pemeriksaan Kualitas Data (Data Quality Check) ---")
        missing_counts = self.raw_df[self.feature_columns].isnull().sum()
        total_missing = missing_counts.sum()
        print(f"Total missing values terdeteksi: {total_missing}")
        for col, count in missing_counts.items():
            if count > 0:
                print(f" - Kolom '{col}': {count} nilai hilang ({count/len(self.raw_df)*100:.1f}%)")
                
        # Salin untuk dibersihkan
        self.clean_df = self.raw_df.copy()
        
        # Penanganan missing values pada fitur
        self.clean_df[self.feature_columns] = self.imputer.fit_transform(self.clean_df[self.feature_columns])
        print("[OK] Missing values berhasil diimputasi menggunakan strategi Median Imputation.")
        
        # Cek distribusi target klasifikasi
        target_counts = self.clean_df["target"].value_counts()
        print(f"Distribusi Kelas Target Klasifikasi (0: Sehat, 1: Berisiko Penyakit Jantung):")
        print(f" - Kelas 0 (Sehat): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.clean_df)*100:.1f}%)")
        print(f" - Kelas 1 (Risiko): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.clean_df)*100:.1f}%)")
        
        return self.clean_df

    def split_and_scale(self):
        """
        Tahap 2: Pembagian Data menjadi Training (80%) dan Testing (20%)
        disertai Stratifikasi kelas dan Standarisasi Fitur (StandardScaler).
        """
        if self.clean_df is None:
            self.inspect_and_clean()
            
        X = self.clean_df[self.feature_columns].values
        y_class = self.clean_df["target"].values
        y_reg = self.clean_df["risk_severity_score"].values
        
        # Train-Test Split untuk Klasifikasi (dengan stratify agar proporsi kelas seimbang)
        X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_class
        )
        
        # Standarisasi fitur: Fit hanya pada data Train untuk mencegah Data Leakage!
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n[TAHAP 2] Pembagian Data (Train/Test Split):")
        print(f" - Data Latih (Training): {X_train.shape[0]} sampel ({(1 - self.test_size)*100:.0f}%)")
        print(f" - Data Uji (Testing)   : {X_test.shape[0]} sampel ({self.test_size*100:.0f}%)")
        print(f" - Jumlah Fitur Prediktor: {X_train.shape[1]}")
        print("[OK] Standarisasi fitur selesai menggunakan StandardScaler (Bebas Data Leakage).")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train_cls": y_train_cls,
            "y_test_cls": y_test_cls,
            "y_train_reg": y_train_reg,
            "y_test_reg": y_test_reg,
            "feature_names": self.feature_columns
        }
