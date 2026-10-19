"""
ethics_audit.py
Analisis Tantangan, Kualitas Data, dan Etika AI:
- Deteksi Bias Algoritmik dan Fairness Audit (Perbandingan performa antar sub-populasi Gender & Usia)
- Transparansi & Explainability (Tingkat kepentingan fitur / Feature Importance)
- Prinsip Etika Medis AI: Keamanan, Privasi Data, dan Pengawasan Manusia (Human-in-the-Loop)

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score

class EthicsAndFairnessAudit:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def audit_gender_fairness(self, X_test, y_test, X_test_unscaled):
        """
        Pemeriksaan Fairness antar Gender (Pria vs Wanita):
        Memastikan model tidak menghasilkan bias atau disparitas tingkat recall
        terhadap pasien wanita dibandingkan pria.
        """
        print("\n" + "="*70)
        print("[ETIKA & TATA KELOLA AI] AUDIT KEADILAN (FAIRNESS & BIAS CHECK)")
        print("="*70)
        
        sex_idx = self.feature_names.index("sex")
        is_male = X_test_unscaled[:, sex_idx] == 1
        is_female = X_test_unscaled[:, sex_idx] == 0
        
        # Prediksi
        y_pred = self.model.predict(X_test)
        
        # Metrik untuk sub-populasi Pria
        acc_male = accuracy_score(y_test[is_male], y_pred[is_male])
        rec_male = recall_score(y_test[is_male], y_pred[is_male], zero_division=0)
        
        # Metrik untuk sub-populasi Wanita
        acc_female = accuracy_score(y_test[is_female], y_pred[is_female])
        rec_female = recall_score(y_test[is_female], y_pred[is_female], zero_division=0)
        
        # Disparate Impact / Fairness Ratio
        disparity_ratio = (rec_female / rec_male) if rec_male > 0 else 1.0
        
        print(f"Sub-populasi Pasien Pria   (n={np.sum(is_male)}): Akurasi = {acc_male*100:.2f}%, Recall = {rec_male*100:.2f}%")
        print(f"Sub-populasi Pasien Wanita (n={np.sum(is_female)}): Akurasi = {acc_female*100:.2f}%, Recall = {rec_female*100:.2f}%")
        print(f" -> Rasio Keadilan Sensitivitas (Female/Male Recall Ratio): {disparity_ratio:.3f}")
        
        if 0.80 <= disparity_ratio <= 1.25:
            print(" [STATUS FAIRNESS] LULUS: Model memenuhi aturan Four-Fifths Rule (Keadilan Terjaga).")
        else:
            print(" [PERINGATAN BIAS] Model menunjukkan disparitas deteksi antar gender. Perlu augmentasi data seimbang.")
            
        return {
            "acc_male": acc_male, "rec_male": rec_male,
            "acc_female": acc_female, "rec_female": rec_female,
            "fairness_ratio": disparity_ratio
        }

    def explain_feature_importance(self):
        """
        Transparansi Model: Menjelaskan fitur klinis paling berpengaruh
        dalam pembuatan keputusan prediksi (Explainable AI dasar).
        """
        print("\n--- Transparansi & Explainability Model (Feature Importance) ---")
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
            for rank, (feat, score) in enumerate(feat_imp, 1):
                bar = "#" * int(score * 40)
                print(f" {rank}. {feat:<20} : {score*100:5.2f}% | {bar}")
        elif hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            feat_imp = sorted(zip(self.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
            for rank, (feat, weight) in enumerate(feat_imp, 1):
                print(f" {rank}. {feat:<20} : Bobot Koefisien = {weight:+.4f}")

    @staticmethod
    def print_governance_guidelines():
        """Panduan Tata Kelola & Etika AI sesuai materi narasumber."""
        print("\n" + "="*70)
        print("[PANDUAN TATA KELOLA & ETIKA AI MEDIS]")
        print("="*70)
        guidelines = [
            "1. Pengawasan Manusia (Human-in-the-Loop): Model AI bertindak sebagai Clinical Decision Support System (CDSS), diagnosis akhir tetap berada pada kewenangan dokter spesialis.",
            "2. Privasi Data Pasien: Data medis harus dianonimisasi sesuai regulasi privasi data (misal UU PDP / GDPR / HIPAA) sebelum proses training.",
            "3. Mitigasi Bias: Distribusi dataset pelatihan harus mewakili keragaman demografis untuk menghindari kesalahan diagnosis pada kelompok minoritas.",
            "4. Audit Berkelanjutan: Model harus dipantau secara berkala terhadap fenomena data drift dan degradasi performa di lingkungan klinis nyata."
        ]
        for g in guidelines:
            print(f" - {g}")
