"""
unsupervised_models.py
Demonstrasi Unsupervised Learning:
- Pengelompokan Data Tanpa Label (Unsupervised Clustering) menggunakan K-Means
- Menemukan pola / profil risiko pasien tersembunyi secara otonom

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning (Supervised vs Unsupervised)
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class UnsupervisedClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = None
        self.silhouette_avg = None

    def fit_and_cluster(self, X_scaled):
        """Melakukan segmentasi data pasien tanpa menggunakan target label apa pun."""
        print(f"\n[UNSUPERVISED LEARNING] Melakukan K-Means Clustering (k={self.n_clusters})...")
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.silhouette_avg = silhouette_score(X_scaled, self.cluster_labels)
        print(f" -> K-Means berhasil menemukan {self.n_clusters} klaster alami.")
        print(f" -> Evaluasi Kualitas Klaster (Silhouette Score): {self.silhouette_avg:.4f} (Mendekati 1 = Klaster terpisah dengan baik)")
        return self.cluster_labels, self.silhouette_avg

    def visualize_clusters(self, X_original, feature_names, output_path="outputs/clustering_visualization.png"):
        """Visualisasi segmentasi pasien berdasarkan fitur klinis utama (misal: Usia vs Kolesterol)."""
        if self.cluster_labels is None:
            raise ValueError("Jalankan fit_and_cluster terlebih dahulu.")
            
        # Ambil indeks usia (age) dan kolesterol (cholesterol)
        age_idx = feature_names.index("age")
        chol_idx = feature_names.index("cholesterol")
        
        plt.figure(figsize=(9, 6))
        scatter = plt.scatter(
            X_original[:, age_idx],
            X_original[:, chol_idx],
            c=self.cluster_labels,
            cmap="viridis",
            alpha=0.7,
            edgecolors="k",
            s=45
        )
        plt.title(f"Unsupervised Learning: Segmentasi Pasien K-Means (k={self.n_clusters})\n(Silhouette Score: {self.silhouette_avg:.3f})", fontsize=13, fontweight="bold")
        plt.xlabel("Usia Pasien (Tahun)", fontsize=11)
        plt.ylabel("Kadar Kolesterol Serum (mg/dl)", fontsize=11)
        plt.colorbar(scatter, label="Klaster Pasien")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"[VISUALISASI] Grafik Klaster disimpan ke: {output_path}")
