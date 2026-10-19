"""
main.py
Eksekusi End-to-End Studi Kasus Machine Learning:
"Sistem Cerdas Deteksi Dini & Stratifikasi Risiko Penyakit Jantung"

Berdasarkan Materi Sesi: Introduction to Machine Learning
Narasumber: Dr. Taniza Tajuddin (Universiti Teknologi MARA Cawangan Kedah, Malaysia)

Tahapan Lengkap:
1. Pengumpulan & Persiapan Data (Data Preparation & Cleaning)
2. Pembagian Data (Train/Test Split & Scaling)
3. Pemilihan Algoritma (Supervised & Unsupervised)
4. Pelatihan Model (Training Phase)
5. Evaluasi Komprehensif (Akurasi, Precision, Recall, F1, MAE, R²)
6. Analisis Overfitting vs Underfitting
7. Demonstrasi Unsupervised Learning (K-Means Clustering)
8. Audit Etika AI, Bias, dan Transparansi
9. Penerapan Model / Inferensi Klinis (Model Deployment & Inference)
"""

import os
import sys

# Tambahkan root path ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_heart_disease_dataset
from src.data_preprocessing import MLDataPipeline
from src.supervised_models import SupervisedModelManager
from src.unsupervised_models import UnsupervisedClustering
from src.model_evaluation import ModelEvaluator
from src.overfitting_analysis import OverfittingExperiment
from src.ethics_audit import EthicsAndFairnessAudit
from src.inference import ClinicalInferenceEngine

def main():
    print("=" * 80)
    print("  STUDI KASUS MACHINE LEARNING: DETEKSI DINI & PREDIKSI RISIKO PENYAKIT JANTUNG")
    print("  Disusun berdasarkan materi dari: Dr. Taniza Tajuddin")
    print("  Universiti Teknologi MARA (UiTM) Cawangan Kedah, Malaysia")
    print("=" * 80)
    
    # 0. Pastikan data tersedia
    data_path = os.path.join("data", "heart_disease_risk.csv")
    if not os.path.exists(data_path):
        print("\n[INFO] Menyiapkan dataset medis...")
        generate_heart_disease_dataset(n_samples=1200, output_path=data_path)
        
    # 1. Tahap 1 & 2: Preprocessing dan Pembagian Data
    pipeline = MLDataPipeline(data_path=data_path, test_size=0.20, random_state=42)
    pipeline.load_data()
    pipeline.inspect_and_clean()
    data_split = pipeline.split_and_scale()
    
    X_train_scaled = data_split["X_train_scaled"]
    X_test_scaled = data_split["X_test_scaled"]
    X_train_unscaled = data_split["X_train"]
    X_test_unscaled = data_split["X_test"]
    y_train_cls = data_split["y_train_cls"]
    y_test_cls = data_split["y_test_cls"]
    y_train_reg = data_split["y_train_reg"]
    y_test_reg = data_split["y_test_reg"]
    feature_names = data_split["feature_names"]
    
    # 2. Tahap 3 & 4: Pemilihan Algoritma & Pelatihan Supervised Learning
    model_mgr = SupervisedModelManager(random_state=42)
    
    # Model Klasifikasi
    trained_classifiers = model_mgr.train_classifiers(X_train_scaled, y_train_cls)
    
    # Model Regresi
    trained_regressors = model_mgr.train_regressors(X_train_scaled, y_train_reg)
    
    # 3. Tahap 5: Evaluasi Model
    evaluator = ModelEvaluator()
    evaluator.evaluate_classifiers(trained_classifiers, X_test_scaled, y_test_cls)
    evaluator.evaluate_regressors(trained_regressors, X_test_scaled, y_test_reg)
    
    # Visualisasi Matriks Evaluasi
    evaluator.plot_confusion_matrices(trained_classifiers, X_test_scaled, y_test_cls)
    evaluator.plot_comparison()
    
    # 4. Eksperimen Khusus: Overfitting vs Underfitting (Decision Tree Depth)
    overfit_exp = OverfittingExperiment(depths=range(1, 16), random_state=42)
    overfit_exp.run_depth_experiment(X_train_scaled, y_train_cls, X_test_scaled, y_test_cls)
    overfit_exp.plot_curve()
    
    # 5. Demonstrasi Unsupervised Learning (K-Means Clustering)
    unsupervised = UnsupervisedClustering(n_clusters=3, random_state=42)
    unsupervised.fit_and_cluster(X_train_scaled)
    unsupervised.visualize_clusters(X_train_unscaled, feature_names)
    
    # 6. Audit Keadilan, Transparansi, & Etika AI
    best_cls_name = "Random Forest"
    best_classifier = trained_classifiers[best_cls_name]
    ethics = EthicsAndFairnessAudit(best_classifier, feature_names)
    ethics.audit_gender_fairness(X_test_scaled, y_test_cls, X_test_unscaled)
    ethics.explain_feature_importance()
    ethics.print_governance_guidelines()
    
    # 7. Tahap 6: Penerapan Model / Inferensi Klinis (Simulasi Pasien Baru)
    best_regressor = trained_regressors["Random Forest Regressor"]
    inference_engine = ClinicalInferenceEngine(
        classifier_model=best_classifier,
        regressor_model=best_regressor,
        scaler=pipeline.scaler,
        feature_names=feature_names
    )
    
    print("\n" + "="*70)
    print("[TAHAP 6] SIMULASI PENERAPAN MODEL (INFERENCE PADA PASIEN BARU)")
    print("="*70)
    
    # Pasien 1: Profil Risiko Tinggi
    patient_high_risk = {
        "age": 62,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp": 155,
        "cholesterol": 310,
        "fasting_blood_sugar": 1,
        "max_heart_rate": 115,
        "exercise_angina": 1,
        "oldpeak": 2.8
    }
    
    # Pasien 2: Profil Risiko Rendah / Sehat
    patient_low_risk = {
        "age": 36,
        "sex": 0,
        "chest_pain_type": 0,
        "resting_bp": 118,
        "cholesterol": 185,
        "fasting_blood_sugar": 0,
        "max_heart_rate": 178,
        "exercise_angina": 0,
        "oldpeak": 0.2
    }
    
    print("\n--- Kasus 1: Pasien A (Pria, 62 Tahun, Gejala Kardiovaskular Signifikan) ---")
    res_a = inference_engine.predict_patient(patient_high_risk)
    for k, v in res_a.items():
        print(f"  * {k:<25}: {v}")
        
    print("\n--- Kasus 2: Pasien B (Wanita, 36 Tahun, Pola Fisiologis Normal) ---")
    res_b = inference_engine.predict_patient(patient_low_risk)
    for k, v in res_b.items():
        print(f"  * {k:<25}: {v}")
        
    print("\n" + "="*80)
    print(" [SELESAI] Seluruh tahapan Machine Learning berhasil dieksekusi dengan sempurna!")
    print(" Artefak visualisasi telah tersimpan di direktori 'outputs/'.")
    print("="*80)

if __name__ == "__main__":
    main()
