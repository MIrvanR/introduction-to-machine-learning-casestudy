"""
supervised_models.py
Tahap 3 & 4 dalam Siklus Machine Learning:
- Pemilihan Algoritma (Algorithm Selection)
- Pelatihan Model (Model Training) untuk Supervised Learning

Mencakup:
1. Supervised Classification (Prediksi Pasien Berisiko / Tidak)
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
2. Supervised Regression (Estimasi Skor Tingkat Keparahan Risiko 0-100)
   - Linear Regression
   - Random Forest Regressor

Materi: Dr. Taniza Tajuddin - Introduction to Machine Learning
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class SupervisedModelManager:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classification_models = {}
        self.regression_models = {}
        
    def init_classification_models(self):
        """Inisialisasi algoritma klasifikasi yang mewakili paradigma linier dan non-linier."""
        self.classification_models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state)
        }
        return self.classification_models

    def train_classifiers(self, X_train, y_train):
        """Melatih semua model klasifikasi terpilih pada data latih."""
        if not self.classification_models:
            self.init_classification_models()
            
        print("\n[TAHAP 3 & 4] Pelatihan Model Supervised Classification:")
        trained_models = {}
        for name, model in self.classification_models.items():
            print(f" -> Melatih algoritma: {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"    [Berhasil] Model {name} telah mempelajari pola dari data latih.")
        return trained_models

    def init_regression_models(self):
        """Inisialisasi algoritma regresi untuk estimasi skor risiko kontinu."""
        self.regression_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=self.random_state)
        }
        return self.regression_models

    def train_regressors(self, X_train, y_train):
        """Melatih semua model regresi pada data latih."""
        if not self.regression_models:
            self.init_regression_models()
            
        print("\n[TAHAP 3 & 4] Pelatihan Model Supervised Regression (Estimasi Skor Risiko):")
        trained_models = {}
        for name, model in self.regression_models.items():
            print(f" -> Melatih algoritma regresi: {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"    [Berhasil] Model {name} siap memprediksi nilai kontinu.")
        return trained_models
