"""
overfitting_analysis.py
Eksperimen Mendalam Isu Overfitting dan Underfitting:
- Underfitting (High Bias): Model terlalu sederhana, gagal menangkap pola data latih dan data uji.
- Overfitting (High Variance): Model terlalu menghafal data latih, gagal generalisasi ke data baru.
- Optimal Balance (Good Fit): Menemukan trade-off bias-variance terbaik.

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class OverfittingExperiment:
    def __init__(self, depths=range(1, 16), random_state=42):
        self.depths = list(depths)
        self.random_state = random_state
        self.train_scores = []
        self.test_scores = []

    def run_depth_experiment(self, X_train, y_train, X_test, y_test):
        """Menguji dampak kedalaman pohon (Tree Depth) terhadap resiko Overfitting & Underfitting."""
        print("\n" + "="*70)
        print("[ANALISIS] EKSPERIMEN OVERFITTING VS UNDERFITTING (DECISION TREE)")
        print("="*70)
        
        self.train_scores = []
        self.test_scores = []
        
        for depth in self.depths:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
            clf.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            self.train_scores.append(train_acc)
            self.test_scores.append(test_acc)
            
            status = ""
            if depth == 1:
                status = "<- [Underfitting: Model Terlalu Sederhana]"
            elif depth == 4:
                status = "<- [Sweet Spot: Generalisasi Optimal]"
            elif depth >= 10:
                status = "<- [Overfitting: Menghafal Data Latih]"
                
            print(f"Depth {depth:2d} | Train Acc: {train_acc*100:6.2f}% | Test Acc: {test_acc*100:6.2f}% | Gap: {(train_acc - test_acc)*100:5.2f}% {status}")
            
        return self.depths, self.train_scores, self.test_scores

    def plot_curve(self, output_path="outputs/overfitting_underfitting_curve.png"):
        """Plot kurva Bias-Variance Tradeoff (Akurasi Training vs Testing)."""
        if not self.train_scores:
            raise ValueError("Jalankan run_depth_experiment terlebih dahulu.")
            
        plt.figure(figsize=(9, 5.5))
        plt.plot(self.depths, np.array(self.train_scores)*100, "o-", color="#d9534f", label="Training Accuracy (Data Latih)")
        plt.plot(self.depths, np.array(self.test_scores)*100, "s-", color="#0275d8", label="Testing Accuracy (Data Uji)")
        
        # Anotasi zona
        plt.axvspan(1, 2, color="orange", alpha=0.15, label="Zona Underfitting (High Bias)")
        plt.axvspan(3, 5, color="green", alpha=0.15, label="Zona Good Fit (Optimal)")
        plt.axvspan(8, 15, color="red", alpha=0.15, label="Zona Overfitting (High Variance)")
        
        plt.title("Analisis Overfitting vs Underfitting pada Decision Tree\n(Materi Dr. Taniza Tajuddin)", fontsize=12, fontweight="bold")
        plt.xlabel("Tingkat Kedalaman Pohon Keputusan (Max Depth)", fontsize=11)
        plt.ylabel("Akurasi (%)", fontsize=11)
        plt.xticks(self.depths)
        plt.ylim(50, 102)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"\n[VISUALISASI] Kurva Overfitting vs Underfitting disimpan ke: {output_path}")
