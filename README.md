# 🧠 Machine Learning–based Crypto & Market Analysis App

Interaktive Analyseplattform für Kryptowährungen, Indizes, Rohstoffe und ausgewählte Aktien – mit Machine Learning, Clustering und technischen Indikatoren.  
Datenquelle: **yfinance (Yahoo Finance)** → **kein API-Key nötig**.

---

# 🇩🇪 DEUTSCH
---

## 📌 Projektbeschreibung

Dieses Projekt ist mein **Abschlussprojekt im Bereich Data Science / Machine Learning**.  
Ziel: Eine **Streamlit-Webanwendung**, die Finanzmärkte analysiert, Muster erkennt und einfache ML-Vorhersagen liefert – leicht verständlich auch für Nicht-Techniker.

Die App:

- lädt Marktdaten via **yfinance** (öffentliche Daten, kein Token),
- berechnet technische Indikatoren,
- trainiert ML-Modelle,
- visualisiert Marktregime & Muster,
- und bietet ein professionelles Dashboard.

---

## 🎯 Funktionsumfang

### 1. Unterstützte Märkte

**Krypto:** BTC, ETH, BNB, SOL, DOGE  
**Indizes:** S&P 500, Nasdaq100, Dow Jones, DAX, VIX  
**Rohstoffe:** Gold, Silber, Erdgas, Öl (WTI/Brent), Kupfer, Platin, Palladium  
**Aktien:** AAPL, MSFT, TSLA, NVDA, META, AMZN, ASML, SAP.DE usw.

---

### 2. Timeframes & Zeiträume

📌 **Candlestick-Timeframes:**
- 15m, 30m  
- 1h, 4h  
- 1d, 1w, 1M

📌 **Analyse-Zeiträume:**
- Gesamte Historie
- Letzte 30 / 90 Tage
- Letztes Jahr
- Letzte 2 Jahre
- Individueller Zeitraum (Start–Enddatum)

---

### 3. Feature Engineering

Indikatoren:
- SMA / EMA (20/50/200)
- RSI
- MACD
- Bollinger Bänder
- ATR
- Log Returns
- Prozentveränderungen
- Rolling Volatilität

---

### 4. Modelle

📌 **Klassische ML-Modelle**
- RandomForestClassifier  
- Logistic Regression  
- KMeans-Clustering  
- KNN (ähnliche Marktphasen finden)

📌 **Deep Learning (optional)**
- LSTM für Zeitreihen  
- optional GRU / CNN

---

### 5. Streamlit-App – Seiten

1. **Marktübersicht**  
2. **Indikatoren & Features**  
3. **ML-Klassifikation (Up/Down)**  
4. **Cluster & ähnliche Marktphasen**  
5. **Thesen & Nachweise**  
6. **Ausblick / Erweiterungen**

---

### 6. Tech Stack

- Python 3.10+
- yfinance (keine API-Keys)
- pandas, numpy
- scikit-learn
- (optional) tensorflow / pytorch
- streamlit, plotly
- Git LFS für .joblib-Modelle

---

# ——————————————————————————————————————
# 🇬🇧 ENGLISH
# ——————————————————————————————————————

## 📌 Project Description

This project is my **final Data Science / Machine Learning project**.  
It provides an **interactive Streamlit web app** to explore financial markets, detect patterns, and run simple machine-learning predictions.

Data is loaded via **yfinance (Yahoo Finance public data – no personal API key required).**

---

## 🎯 Features

### 1. Supported Markets

**Crypto:** BTC, ETH, BNB, SOL, DOGE  
**Indices:** S&P500, Nasdaq100, Dow Jones, DAX, VIX  
**Commodities:** Gold, Silver, Oil, Natural Gas, Copper, Platinum, Palladium  
**Stocks:** AAPL, MSFT, NVDA, TSLA, META, AMZN, ASML, SAP.DE, etc.

---

### 2. Timeframes & Date Ranges

📌 **Candlestick Timeframes:**
- 15m, 30m  
- 1h, 4h  
- 1d, 1w, 1M  

📌 **Date Range Filters:**
- Full historical data  
- Last 30 / 90 days  
- Last 1 / 2 years  
- Custom range (start–end date)

---

### 3. Feature Engineering

Indicators:
- SMA / EMA (20/50/200)
- RSI
- MACD
- Bollinger Bands
- ATR
- Log returns
- Percentage changes
- Rolling volatility

---

### 4. Models

📌 **Classical ML**
- RandomForestClassifier  
- Logistic Regression  
- KMeans clustering  
- KNN similarity search  

📌 **Deep Learning (optional)**
- LSTM for time series  
- optional GRU / CNN variants

---

### 5. Streamlit App – Pages

1. **Market Overview**  
2. **Indicators & Feature Plots**  
3. **ML Classification (Up/Down prediction)**  
4. **Clusters & Similar Market Phases**  
5. **Hypotheses & Evidence**  
6. **Future Work**

---

### 6. Tech Stack

- Python 3.10+  
- yfinance (no user token)  
- pandas / numpy  
- scikit-learn  
- tensorflow or pytorch (optional)  
- streamlit + plotly  
- Git LFS for model storage  

---

## 🔧 Quickstart

```bash
pip install -r requirements.txt
streamlit run app/main_app.py
