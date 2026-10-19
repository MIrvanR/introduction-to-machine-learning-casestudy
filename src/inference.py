"""
inference.py
Tahap 6 dalam Siklus Machine Learning: Penerapan Model (Model Deployment & Inference)
- Melakukan inferensi / prediksi klinis terhadap data pasien baru yang belum pernah dilihat model sebelumnya.

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

import numpy as np

class ClinicalInferenceEngine:
    def __init__(self, classifier_model, regressor_model, scaler, feature_names):
        self.classifier = classifier_model
        self.regressor = regressor_model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict_patient(self, patient_dict):
        """
        Menerima data pasien baru berupa dictionary dan mengembalikan hasil diagnosis:
        - Kategori Diagnosis (Sehat / Berisiko Penyakit Jantung)
        - Probabilitas Risiko (%)
        - Estimasi Skor Keparahan Risiko (0-100)
        - Rekomendasi Tindakan Klinis
        """
        # Konversi dict ke array sesuai urutan feature_names
        patient_vector = np.array([patient_dict[col] for col in self.feature_names]).reshape(1, -1)
        
        # Standarisasi fitur dengan scaler yang sudah dilatih (Inference pipeline)
        patient_scaled = self.scaler.transform(patient_vector)
        
        # 1. Prediksi Klasifikasi
        pred_class = self.classifier.predict(patient_scaled)[0]
        
        # Hitung probabilitas jika model mendukung
        if hasattr(self.classifier, "predict_proba"):
            pred_proba = self.classifier.predict_proba(patient_scaled)[0][1]
        else:
            pred_proba = float(pred_class)
            
        # 2. Prediksi Regresi (Skor Keparahan)
        pred_severity = self.regressor.predict(patient_scaled)[0]
        
        # 3. Formulasi Keputusan & Rekomendasi
        status = "BERISIKO PENYAKIT JANTUNG" if pred_class == 1 else "SEHAT / RISIKO RENDAH"
        
        recommendation = (
            "Rekomendasi: Rujuk segera ke dokter spesialis jantung untuk pemeriksaan angiografi & EKG lanjutan."
            if pred_class == 1 else
            "Rekomendasi: Lanjutkan gaya hidup sehat, kontrol tekanan darah rutin, dan olahraga berkala."
        )
        
        report = {
            "Diagnosis": status,
            "Kode_Diagnosis": int(pred_class),
            "Probabilitas_Risiko": f"{pred_proba * 100:.2f}%",
            "Skor_Keparahan_Risiko": f"{pred_severity:.1f} / 100",
            "Rekomendasi_Klinis": recommendation
        }
        
        return report
