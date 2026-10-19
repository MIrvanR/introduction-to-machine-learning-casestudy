"""
generate_data.py
Generator dataset sintetis medis untuk studi kasus Machine Learning:
Prediksi dan Stratifikasi Risiko Penyakit Jantung (Cardiovascular Disease Risk Prediction).

Materi Referensi: Sesi 'Introduction to Machine Learning' bersama Dr. Taniza Tajuddin
(Universiti Teknologi MARA Cawangan Kedah, Malaysia).
"""

import numpy as np
import pandas as pd
import os

def generate_heart_disease_dataset(n_samples=1200, random_state=42, output_path="data/heart_disease_risk.csv"):
    np.random.seed(random_state)
    
    # 1. Fitur Demografis & Fisiologis
    age = np.random.normal(loc=54, scale=9.5, size=n_samples).astype(int)
    age = np.clip(age, 29, 80)
    
    # 65% Pria (1), 35% Wanita (0)
    sex = np.random.binomial(n=1, p=0.65, size=n_samples)
    
    # Jenis nyeri dada (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)
    chest_pain_type = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.48, 0.17, 0.28, 0.07])
    
    # Tekanan darah istirahat (mm Hg)
    resting_bp = np.random.normal(loc=131, scale=17.5, size=n_samples).astype(int)
    resting_bp = np.clip(resting_bp, 94, 200)
    
    # Kolesterol serum (mg/dl)
    cholesterol = np.random.normal(loc=246, scale=51.8, size=n_samples).astype(int)
    cholesterol = np.clip(cholesterol, 126, 560)
    
    # Gula darah puasa > 120 mg/dl (1: True, 0: False)
    fasting_blood_sugar = np.random.binomial(n=1, p=0.15, size=n_samples)
    
    # Detak jantung maksimum dicapai (Thalach)
    # Biasanya berkurang seiring bertambahnya usia: ~220 - age dengan variansi
    max_heart_rate = (205 - 0.7 * age + np.random.normal(0, 14, size=n_samples)).astype(int)
    max_heart_rate = np.clip(max_heart_rate, 70, 205)
    
    # Angina akibat aktivitas olahraga (1: Ya, 0: Tidak)
    exercise_angina = np.random.binomial(n=1, p=0.32, size=n_samples)
    
    # Depresi ST (oldpeak)
    oldpeak = np.abs(np.random.exponential(scale=0.9, size=n_samples))
    oldpeak = np.round(np.clip(oldpeak, 0.0, 6.2), 2)
    
    # 2. Formulasi Logit Klinis (Probabilitas Penyakit Jantung)
    # Mencerminkan pola biologis sebenarnya
    logit = (
        -2.2
        + 0.045 * (age - 50)
        + 0.65 * sex
        + 0.70 * (chest_pain_type > 0)
        + 0.015 * (resting_bp - 120)
        + 0.008 * (cholesterol - 200)
        + 0.50 * fasting_blood_sugar
        - 0.025 * (max_heart_rate - 140)
        + 0.85 * exercise_angina
        + 0.60 * oldpeak
    )
    
    # Sigmoid function untuk probabilitas
    prob = 1 / (1 + np.exp(-logit))
    
    # Target klasifikasi biner (1: Memiliki Penyakit Jantung, 0: Tidak)
    target = (prob > 0.50).astype(int)
    
    # Target regresi kontinu: Risk Severity Score (skala 0 - 100)
    # Kombinasi probabilitas dasar ditambah variansi noise klinis
    risk_score = prob * 85 + (age / 80) * 10 + np.random.normal(0, 3, size=n_samples)
    risk_score = np.round(np.clip(risk_score, 5.0, 99.5), 2)
    
    # 3. Membuat DataFrame
    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "risk_severity_score": risk_score,
        "target": target
    })
    
    # 4. Mensimulasikan kondisi dunia nyata: sedikit missing values untuk tahap Preprocessing
    # Sekitar 2.5% baris memiliki nilai hilang pada cholesterol dan resting_bp
    missing_indices_chol = np.random.choice(df.index, size=int(0.025 * n_samples), replace=False)
    missing_indices_bp = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    df.loc[missing_indices_chol, "cholesterol"] = np.nan
    df.loc[missing_indices_bp, "resting_bp"] = np.nan
    
    # Pastikan direktori output tersedia
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset berhasil digenerate: {output_path} ({n_samples} baris, {df.shape[1]} kolom)")
    return df

if __name__ == "__main__":
    generate_heart_disease_dataset()
