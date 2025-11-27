import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Liste aller Märkte (Version B + C)
MARKETS = {
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "DOGE-USD"],
    "Indices": ["^GSPC", "^NDX", "^DJI", "^GDAXI", "^VIX"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F"],
    "Stocks": ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "ASML.AS", "SAP.DE"]
}

def download_data(ticker, period="max", interval="1d"):
    """
    Lade historische Marktdaten über yfinance.
    Kein API-Key nötig.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Fehler beim Laden von {ticker}: {e}")
        return None


def save_data(df, ticker, folder="data/raw"):
    """
    Speichere Rohdaten lokal als CSV.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{ticker}.csv")
    df.to_csv(path)
    print(f"✅ Gespeichert: {path}")


def fetch_all_markets(period="max", interval="1d"):
    """
    Lade ALLE Märkte (Crypto, Indizes, Rohstoffe, Aktien)
    und speichere sie in data/raw/
    """
    for category, tickers in MARKETS.items():
        print(f"\n📂 Kategorie: {category}")

        for ticker in tickers:
            print(f"⬇️ Lade {ticker}...")
            df = download_data(ticker, period, interval)

            if df is not None:
                save_data(df, ticker)


if __name__ == "__main__":
    print("🚀 Starte Daten-Download...")
    fetch_all_markets()
    print("🎉 Fertig!")
 