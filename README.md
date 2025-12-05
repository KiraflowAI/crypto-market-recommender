# ₿ Crypto Market Recommender  
_Ein interaktives Streamlit-Dashboard zur Analyse, Clusterbildung & ML-Vorhersage von Bitcoin-Marktphasen_

Dieses Projekt untersucht den Bitcoin-Markt mithilfe explorativer Datenanalyse, Machine Learning (Clustering, Klassifikation, Regression) und statistischer Methoden.  
Alle Modelle liegen **fertig trainiert** im Repository – die App kann **ohne erneutes Training** direkt gestartet werden.

---

## 🚀 Features der Streamlit-App

### **1. Markt-Regime Analyse (Bull / Sideways / Bear)**
- Identifikation ökonomischer Marktphasen  
- Häufigkeiten & statistische Kennzahlen pro Regime  
- Rendite-, Volatilitäts- & Momentum-Auswertungen  
- MA50–MA200 Trendindikator zur strukturellen Regimebestätigung  

### **2. ML-Clustering (KMeans & Vergleichsmodelle)**
- KMeans als Hauptmodell zur Marktphasen-Erkennung  
- Alternative Modelle: MiniBatchKMeans, GMM, VBGMM, Spectral  
- PCA-2D-Visualisierung aller Tage  
- Interpretation der Cluster (Returns, Volatilität, Momentum)

### **3. Preisvorhersage (Regression)**
- Modelle: Linear, Ridge, Lasso, ElasticNet, RandomForest  
- Analyse der Modellgüte (RMSE, R²)  
- Prognose der täglichen Preisveränderung  

### **4. Up/Down-Vorhersage (Direction Classification)**
- Modelle: Logistic Regression, SVM, KNN, RandomForest, GradientBoosting  
- F1-Scores & Performancevergleich  
- Visualisierung der Grenzen kurzfristiger Trendvorhersagen  

### **5. Explorative Datenanalyse (EDA)**
- Return-Histogramme & Ausreißer  
- Volatilitätsanalyse  
- Korrelationsmatrix  
- Zeitreihen (Preis & Volatilität)  
- Häufigkeiten der Regime & Cluster  

---

## 📁 Projektstruktur

crypto-market-recommender/
│
├── app/
│ └── btc_dashboard.py # Haupt-Streamlit-App
│
├── data/
│ ├── raw/ # Ungereinigte historische Daten (BTC + Indizes)
│ └── processed/ # Alle verarbeiteten CSVs & Modell-Outputs
│ ├── btc_clean.csv
│ ├── btc_view.csv
│ ├── btc_master_view_final.csv
│ ├── btc_clusters.csv
│ ├── btc_clusters_pca.csv
│ ├── price_daily_model_metrics.csv
│ ├── clustering_metrics.csv
│ └── clustering_labels_all_models.joblib
│
├── models/
│ ├── clustering/ # KMeans, GMM, VB-GMM, MiniBatchKMeans + Scaler
│ ├── direction/ # Klassifikationsmodelle + direction_model_metrics.csv
│ └── price_daily/ # Regressionsmodelle (1d, 7d, 30d, 90d, 365d)
│
├── notebooks/ # Reproduzierbare Jupyter-Notebooks
│ ├── 01_explore_data.ipynb
│ ├── 02_classification_direction.ipynb
│ ├── 03_clustering_market_regimes.ipynb
│ ├── 04_regression_price.ipynb
│ └── 05_master_view.ipynb
│
├── scripts/
│ ├── fetch_data.py # (Optional) Rohdatenabruf
│ └── src/
│
├── requirements.txt
└── README.md


---

## ⚙️ Voraussetzungen

- **Python 3.10 – 3.11**  
- Git  
- Virtuelle Umgebung empfohlen (venv)

---

## 🚀 Quickstart – Projekt starten

<!-- ```bash
# 1) Repository klonen
git clone https://github.com/KiraflowAI/crypto-market-recommender.git
cd crypto-market-recommender

# 2) Virtuelle Umgebung erstellen
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 3) Pip aktualisieren
python -m pip install --upgrade pip

# 4) Dependencies installieren
pip install -r requirements.txt

# 5) Streamlit starten
streamlit run app/btc_dashboard.py -->

Die App startet auf:

👉 http://localhost:8501

📊 Datenquellen

Dieses Projekt nutzt frei verfügbare historische Daten, u. a.:

Bitcoin OHLCV (1D)

VIX Index

S&P500 (^GSPC)

NASDAQ (^NDX)

DAX (^GDAXI)

Dow Jones (^DJI)

Keine API-Keys erforderlich.

📊 Datenquellen

Dieses Projekt nutzt frei verfügbare historische Daten, u. a.:

Bitcoin OHLCV (1D)

Keine API-Keys erforderlich.

🧠 Modelle im Projekt
Clustering

KMeans — Hauptmodell (Cluster_3)

GMM, VBGMM, MiniBatchKMeans, Spectral — Vergleichsmodelle

PCA (2D) für Visualisierungen

Direction Classification

Logistic Regression

Support Vector Machine (SVM)

KNN Classifier

RandomForestClassifier

GradientBoostingClassifier
→ Alle Modelle werden mit F1-Score verglichen.

Price Regression

Linear Regression

Ridge

Lasso

ElasticNet

RandomForest Regressor
→ Metriken: RMSE, MAE, R²

Alle trainierten Modelle liegen im Repository unter:
models/

🧪 Reproduzierbarkeit

Alle Schritte sind dokumentiert in:

01_explore_data.ipynb

02_classification_direction.ipynb

03_clustering_market_regimes.ipynb

04_regression_price.ipynb

05_master_view.ipynb

Diese Notebooks erzeugen exakt dieselben Dateien, die die Streamlit-App später nutzt.

❗ Hinweise

Dieses Projekt dient Bildungs- und Analysezwecken.
Es ist nicht zur finanziellen Entscheidungsfindung gedacht.