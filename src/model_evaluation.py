"""
model_evaluation.py
Tahap 5 dalam Siklus Machine Learning: Evaluasi Model (Model Evaluation)
- Klasifikasi: Akurasi, Precision, Recall, F1-Score, Confusion Matrix
- Regresi: Mean Absolute Error (MAE), R² (Koefisien Determinasi)

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, r2_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self):
        self.classification_results = []
        self.regression_results = []

    def evaluate_classifiers(self, trained_models, X_test, y_test):
        """
        Mengevaluasi kinerja model klasifikasi pada data uji (Unseen Data).
        Menghitung Akurasi, Presisi, Recall, dan F1-Score.
        """
        print("\n" + "="*70)
        print("[TAHAP 5] HASIL EVALUASI MODEL KLASIFIKASI (DATA TESTING)")
        print("="*70)
        
        results = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "Confusion_Matrix": cm
            })
            
            print(f"\nAlgoritma: {name}")
            print(f" -> Accuracy : {acc*100:.2f}% (Tingkat prediksi benar keseluruhan)")
            print(f" -> Precision: {prec*100:.2f}% (Ketepatan deteksi positif)")
            print(f" -> Recall   : {rec*100:.2f}% (Sensitivitas deteksi pasien berisiko)")
            print(f" -> F1-Score : {f1*100:.2f}% (Harmonic mean presisi & recall)")
            print(f" -> Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
            
        self.classification_results = results
        return pd.DataFrame(results).drop(columns=["Confusion_Matrix"])

    def evaluate_regressors(self, trained_models, X_test, y_test):
        """
        Mengevaluasi model regresi untuk memprediksi nilai kontinu skor risiko.
        Menghitung MAE dan R².
        """
        print("\n" + "="*70)
        print("[TAHAP 5] HASIL EVALUASI MODEL REGRESI (DATA TESTING)")
        print("="*70)
        
        results = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                "Model": name,
                "MAE": mae,
                "R2_Score": r2
            })
            
            print(f"\nAlgoritma Regresi: {name}")
            print(f" -> MAE (Mean Absolute Error): {mae:.2f} (Rata-rata deviasi skor risiko)")
            print(f" -> R² Score (Determinasi)   : {r2*100:.2f}% (Proporsi variansi data yang dijelaskan)")
            
        self.regression_results = results
        return pd.DataFrame(results)

    def plot_confusion_matrices(self, trained_models, X_test, y_test, output_path="outputs/confusion_matrix.png"):
        """Plotting visualisasi Confusion Matrix untuk semua classifier."""
        n_models = len(trained_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]
            
        for ax, (name, model) in zip(axes, trained_models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"Confusion Matrix:\n{name}", fontsize=11, fontweight="bold")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(["Sehat (0)", "Risiko (1)"])
            ax.set_yticklabels(["Sehat (0)", "Risiko (1)"])
            
            # Label angka di dalam kotak
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight="bold", fontsize=12)
                    
            ax.set_ylabel("Label Aktual (True Label)", fontsize=10)
            ax.set_xlabel("Label Prediksi (Predicted)", fontsize=10)
            
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"\n[VISUALISASI] Confusion Matrix disimpan ke: {output_path}")

    def plot_comparison(self, output_path="outputs/model_comparison.png"):
        """Visualisasi perbandingan metrik performa klasifikasi."""
        if not self.classification_results:
            return
            
        df = pd.DataFrame(self.classification_results).drop(columns=["Confusion_Matrix"])
        models = df["Model"]
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        x = np.arange(len(models))
        width = 0.2
        
        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + (i - 1.5) * width, df[metric] * 100, width, label=metric)
            
        plt.xlabel("Algoritma Machine Learning", fontsize=11, fontweight="bold")
        plt.ylabel("Skor Persentase (%)", fontsize=11, fontweight="bold")
        plt.title("Perbandingan Metrik Evaluasi Model Klasifikasi\n(Dr. Taniza Tajuddin - Introduction to ML)", fontsize=13, fontweight="bold")
        plt.xticks(x, models, fontsize=10)
        plt.ylim(0, 105)
        plt.legend(loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"[VISUALISASI] Perbandingan Model disimpan ke: {output_path}")
