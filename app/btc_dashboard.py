import streamlit as st
import pandas as pd
import os
import joblib 
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ==================================================================================================
# 1. STREAMLIT KONFIGURATION
# ==================================================================================================
st.set_page_config(layout="wide", page_title="₿ BTC Markt-Regime & Prognose-Dashboard") 

# ==================================================================================================
# 2. DATENPFADE & LADEFUNKTIONEN (🚨 WICHTIG: PFADE ANPASSEN!)
# ==================================================================================================

# # 🛠️ ABSOLUTE PFADE (Basierend auf Ihren Angaben)
# DATA_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/btc_master_view_final.csv'
# PCA_DATA_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/btc_clusters_pca.csv'
# REGRESSION_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/price_daily_model_metrics.csv'
# CLUSTERING_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/clustering_metrics.csv'
# DIRECTION_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/models/direction/direction_model_metrics.csv'


# @st.cache_data
# def load_data(path):
#     """Lädt die Master-Datei und führt erste Bereinigungen durch.
#     Stellt sicher, dass PC1 und PC2 numerisch sind."""
#     st.info(f"📁 Versuche, Master-Datei von Pfad: **{path}** zu laden...")
    
#     if not os.path.exists(path):
#         st.error(f"❌ **Datei nicht gefunden!** Der Pfad existiert nicht: `{path}`")
#         return None
#     try:
#         # HINWEIS: Master-Datei lädt wahrscheinlich mit Semikolon (;) als Separator
#         df = pd.read_csv(path, sep=';', index_col='Date', parse_dates=True, on_bad_lines='skip') 
        
#         required_cols = ['PC1', 'PC2', 'Regime', 'Close']
#         if not all(col in df.columns for col in required_cols):
#              st.error(f"❌ **Fehlende Spalten im Master-DF.** Erwartet: {required_cols}")
#              return None
        
#         # Erzwingen der numerischen Konvertierung für PC1 und PC2 (Fix für den PCA Plot)
#         df['PC1'] = pd.to_numeric(df['PC1'], errors='coerce')
#         df['PC2'] = pd.to_numeric(df['PC2'], errors='coerce')
        
#         # Entferne Zeilen mit fehlenden Werten in kritischen Spalten
#         df.dropna(subset=['PC1', 'PC2', 'Regime'], inplace=True) 
        
#         st.success(f"✅ **Daten erfolgreich geladen und PCA-Spalten als Float konvertiert!** Shape: {df.shape}")
#         return df
#     except Exception as e:
#         st.error(f"⚠️ **Schwerer Fehler beim Einlesen der CSV**: `{e}`")
#         return None

# # NEUE, ROBUSTERE LADEFUNKTION FÜR METRIKEN
# @st.cache_data
# def load_metrics(path, expected_cols=None):
#     """
#     Lädt eine Metrik-Datei und versucht es mit verschiedenen Separatoren, 
#     um Parsing-Fehler zu vermeiden.
#     """
#     if not os.path.exists(path):
#         st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden unter `{path}`.")
#         return None
    
#     separators = [';', ',', '\t']
#     df = None
    
#     for sep in separators:
#         try:
#             df = pd.read_csv(path, sep=sep)
#             # Erfolgreich geparst, wenn mehr als 1 Spalte gefunden
#             if df.shape[1] > 1:
#                 # Prüfen, ob eine erwartete Schlüsselspalte vorhanden ist, falls angegeben
#                 if expected_cols is None or any(col in df.columns for col in expected_cols):
#                     return df
#             # Wenn nur eine Spalte geladen wurde, versuchen wir den nächsten Separator
#         except Exception:
#             continue
            
#     # Nur Fehler anzeigen, wenn das Laden komplett fehlschlägt
#     st.error(f"❌ Metriken konnten nicht geladen oder richtig geparst werden: `{path}`. Versuchte Separatoren: {separators}. Überprüfen Sie das Dateiformat.")
#     return None

# df_master = load_data(DATA_PATH)

# # --- KLASSISCHE DATENLADUNG MIT ROBUSTER FUNKTION ---
# df_cluster_metrics = load_metrics(CLUSTERING_METRICS_PATH, expected_cols=['model', 'Model', 'Silhouette'])
# df_dir_metrics = load_metrics(DIRECTION_METRICS_PATH, expected_cols=['model', 'Model', 'Accuracy'])
# df_reg_metrics = load_metrics(REGRESSION_METRICS_PATH, expected_cols=['model', 'Model', 'horizon', 'Horizont']) 

# ==================================================================================================
# 2. DATENPFADE & LADEFUNKTIONEN (FINAL KORRIGIERT: Trennzeichen-Konflikt gelöst)
# ==================================================================================================

# 🛠️ ABSOLUTE PFADE (Basierend auf Ihren Angaben)
DATA_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/btc_master_view_final.csv'
PCA_DATA_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/btc_clusters_pca.csv'
REGRESSION_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/price_daily_model_metrics.csv'
CLUSTERING_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/data/processed/clustering_metrics.csv'
DIRECTION_METRICS_PATH = '/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/models/direction/direction_model_metrics.csv' # 🚨 Läd mit sep=','

# --- Ladefunktion für die Master-Datei (nutzt Semikolon) ---
@st.cache_data
def load_data(path):
    """Lädt die Master-Datei und führt erste Bereinigungen durch."""
    st.info(f"📁 Versuche, Master-Datei von Pfad: **{path}** zu laden...")
    
    if not os.path.exists(path):
        st.error(f"❌ **Datei nicht gefunden!** Der Pfad existiert nicht: `{path}`")
        return None
    try:
        # ✔ Die Master-Datei verwendet ein Semikolon (;)
        df = pd.read_csv(path, sep=';', index_col='Date', parse_dates=True, on_bad_lines='skip') 
        
        required_cols = ['PC1', 'PC2', 'Regime', 'Close']
        if not all(col in df.columns for col in required_cols):
             st.error(f"❌ **Fehlende Spalten im Master-DF.** Erwartet: {required_cols}")
             return None
        
        # Erzwingen der numerischen Konvertierung (Behält vorhandenen Fix)
        df['PC1'] = pd.to_numeric(df['PC1'], errors='coerce')
        df['PC2'] = pd.to_numeric(df['PC2'], errors='coerce')
        df.dropna(subset=['PC1', 'PC2', 'Regime'], inplace=True) 
        
        st.success(f"✅ **Daten erfolgreich geladen und PCA-Spalten als Float konvertiert!** Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"⚠️ **Schwerer Fehler beim Einlesen der CSV**: `{e}`")
        return None

# --- Ladefunktion für Metriken, die wir KOMMAGETRENNT gespeichert haben (FIX FÜR DIRECTION_METRICS) ---
@st.cache_data
def load_metrics_comma_sep(path):
    """
    Lädt die Klassifikations-Metriken explizit mit Komma-Separator.
    Dies behebt den Fehler beim Parsen von direction_model_metrics.csv.
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden unter `{path}`.")
        return None
    try:
        # ✔ WICHTIG: Erzwingt das Komma-Trennzeichen, das im Jupyter Notebook genutzt wird.
        df = pd.read_csv(path, sep=',')
        return df
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Metriken unter {path}: {e}")
        return None

# --- Ursprüngliche, robuste Ladefunktion (Wird beibehalten, da sie für andere Files funktioniert) ---
@st.cache_data
def load_metrics_robust(path, expected_cols=None):
    """
    Lädt eine Metrik-Datei und versucht es mit verschiedenen Separatoren, 
    um Parsing-Fehler zu vermeiden (wird für Regress./Cluster-Files genutzt).
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden unter `{path}`.")
        return None
    
    # 💡 Wir lassen die ursprüngliche Logik, um andere Metriken zu laden.
    separators = [';', ',', '\t']
    df = None
    
    for sep in separators:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                if expected_cols is None or any(col in df.columns for col in expected_cols):
                    return df
        except Exception:
            continue
            
    st.error(f"❌ Metriken konnten nicht geladen oder richtig geparst werden: `{path}`. Versuchte Separatoren: {separators}. Überprüfen Sie das Dateiformat.")
    return None


# ==================================================================================================
# 3. DATENLADUNG STARTEN
# ==================================================================================================

df_master = load_data(DATA_PATH)

# --- KLASSISCHE DATENLADUNG ---
# Nutzt die robuste Funktion für die Files, bei denen wir den Separator nicht kennen
df_cluster_metrics = load_metrics_robust(CLUSTERING_METRICS_PATH, expected_cols=['model', 'Model', 'Silhouette'])
df_reg_metrics = load_metrics_robust(REGRESSION_METRICS_PATH, expected_cols=['model', 'Model', 'horizon', 'Horizont'])


# --- 🚨 KORRIGIERTE DATENLADUNG FÜR KLASSIFIKATIONS-METRIKEN ---
# Nutzt die neue Funktion, um den Komma-Separator zu erzwingen
df_dir_metrics = load_metrics_comma_sep(DIRECTION_METRICS_PATH)


# --- 🚨 KRITISCHE KORREKTUR DER REGRESSIONS-METRIKEN ---
# Behebt den 'horizon'-Fehler und den 'Modell'-KeyError durch Standardisierung der Spaltennamen
if df_reg_metrics is not None:
    
    # Standardisiere den Modell-Namen auf 'model'
    model_col = next((col for col in ['model', 'Model'] if col in df_reg_metrics.columns), None)
    if model_col is not None:
        df_reg_metrics.rename(columns={model_col: 'model'}, inplace=True)
    else:
        st.error("❌ Schwerer Fehler: Die Spalte 'model' (Modellname) fehlt in den Regressions-Metriken.")
        df_reg_metrics = None # Deaktiviere den Tab bei fehlender Schlüsselspalte

    # Standardisiere den Horizont-Namen auf 'horizon' und konvertiere
    horizon_col = next((col for col in ['horizon', 'Horizont'] if col in df_reg_metrics.columns), None)
    
    if df_reg_metrics is not None and horizon_col is not None:
        try:
            # Versuche, 7 (int) zu "7d" (string) zu konvertieren
            df_reg_metrics['horizon'] = df_reg_metrics[horizon_col].astype(str).str.replace('d', '')
            df_reg_metrics['horizon'] = df_reg_metrics['horizon'].astype(int).astype(str) + 'd'
            
            # Bereinigung der Spalten (wenn nötig)
            if horizon_col != 'horizon':
                 df_reg_metrics.drop(columns=[horizon_col], inplace=True)
                 
            st.success("✅ Regressions-Metriken: 'horizon' Spalte erfolgreich für Filterung konvertiert.")
        except Exception as e:
            st.error(f"⚠️ Fehler bei der Konvertierung der 'horizon'-Spalte in df_reg_metrics: {e}. Prüfen Sie die Werte.")
            df_reg_metrics = None # Deaktiviere den Tab bei fehlerhafter Konvertierung
    elif df_reg_metrics is not None:
        st.error("❌ Schwerer Fehler: Die Spalte 'horizon' fehlt in den geladenen Regressions-Metriken. (Wird ignoriert)")
        df_reg_metrics = None
# -------------------------------------------------------------------------


# ==================================================================================================
# 3. HILFSFUNKTIONEN FÜR DATAFRAME STYLING UND FORMATIERUNG
# ==================================================================================================

# --- STYLING FUNKTIONEN (Jetzt abgesichert mit .get() gegen KeyError) ---

def highlight_selected_reg_model(s, current_model):
    """Hebt das aktuell ausgewählte Regressionsmodell hervor (Spalte 'Modell')."""
    # Fix: Verwende .get() um KeyError zu vermeiden, falls die Spalte fehlt
    return ['background-color: #60A5FA' if s.get('Modell') == current_model else '' for _ in s]

def highlight_selected_dir_model(s, current_model):
    """Hebt das aktuell ausgewählte Klassifikationsmodell hervor (Spalte 'Modell')."""
    # Fix: Verwende .get() um KeyError zu vermeiden, falls die Spalte fehlt
    return ['background-color: #60A5FA' if s.get('Modell') == current_model else '' for _ in s]

def highlight_selected_cluster_model(s, current_model):
    """Hebt das aktuell ausgewählte Clustering-Modell hervor (Spalte 'Modell')."""
    return ['background-color: #60A5FA' if s.get('Modell') == current_model else '' for _ in s]

def highlight_focus_day(s, focus_date_str):
    """Hebt den Fokus-Tag in der Zeile hervor (angewendet auf axis=1)."""
    try:
        current_date_str = s.name.strftime('%Y-%m-%d')
    except AttributeError:
        current_date_str = str(s.name)

    if current_date_str == focus_date_str:
        return ['background-color: #FBBF24; color: black; font-weight: bold;'] * len(s)
    
    return [''] * len(s)

def highlight_return_analogy(s):
    """Hebt positive/negative Returns in der Analogie-Tabelle hervor."""
    # Der Style muss auf die Zelle in der Spalte 'Tatsächlicher 7d Return (Real)' angewendet werden.
    if s.name == 'Tatsächlicher 7d Return (Real)':
        styles = []
        for val_str in s:
            try:
                # Versuche, den Wert als Zahl zu interpretieren (entferne "%" und Leerzeichen)
                val = float(str(val_str).replace('%', '').strip().replace(',', '.'))
                
                if val > 0:
                    styles.append('background-color: rgba(40, 167, 69, 0.4); color: black')
                elif val < 0:
                    styles.append('background-color: rgba(220, 53, 69, 0.4); color: white')
                else:
                    styles.append('')
            except ValueError:
                styles.append('')
        return styles
    # Rückgabe einer leeren Liste von Styles, wenn es nicht die Zielspalte ist
    return [''] * len(s)
    


# --- FORMATIERUNGS FUNKTIONEN (Für saubere Anzeige) ---

def format_prices(df):
    """Formatiert Währungswerte und Kennzahlen für die Anzeige."""
    df_f = df.copy()
    price_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'MA7', 'MA21', 'MA50', 'MA200'] if col in df_f.columns]
    for col in price_cols:
        df_f[col] = pd.to_numeric(df_f[col], errors='coerce').apply(lambda x: f'{x:,.2f}') 
    
    ratio_cols = [col for col in ['Return', 'Volatility30', 'Momentum7'] if col in df_f.columns]
    for col in ratio_cols:
        df_f[col] = pd.to_numeric(df_f[col], errors='coerce').apply(lambda x: f'{x:.4f}')
        
    if 'Volume' in df_f.columns:
        df_f['Volume'] = pd.to_numeric(df_f['Volume'], errors='coerce').apply(lambda x: f'{x:,.0f}')
        
    return df_f

def format_cluster_pca(df):
    """Formatiert PCA-Werte und benennt Cluster-ID um."""
    df_c = df.copy()
    if 'Cluster_3' in df_c.columns:
        df_c.rename(columns={'Cluster_3': 'Cluster_ID'}, inplace=True)
        
    pca_cols = [col for col in ['PC1', 'PC2'] if col in df_c.columns]
    for col in pca_cols:
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce').apply(lambda x: f'{x:.3f}')
        
    return df_c

def format_signals(df):
    """Formatiert Wahrscheinlichkeiten und Preisprognosen."""
    df_s = df.copy()
    
    if 'Prob_Up' in df_s.columns:
        df_s['Prob_Up'] = pd.to_numeric(df_s['Prob_Up'], errors='coerce').apply(lambda x: f'{x:.3f}')
        
    pred_cols = [col for col in ['Pred_1d', 'Pred_7d', 'Pred_30d', 'Pred_90d', 'Pred_365d'] if col in df_s.columns]
    for col in pred_cols:
        df_s[col] = pd.to_numeric(df_s[col], errors='coerce').apply(lambda x: f'{x:,.2f}')
        
    return df_s

# ==================================================================================================
# 4. PLOTLY CHART & ANALOGIE FUNKTIONEN
# ==================================================================================================

def create_candlestick_chart(df):
    """Erstellt einen Plotly Candlestick Chart mit MAs und Regime-Hintergrund."""
    
    REGIME_COLORS = {
        'Bull': 'rgba(40, 167, 69, 0.15)',      
        'Bear': 'rgba(220, 53, 69, 0.15)',      
        'Sideways': 'rgba(255, 193, 7, 0.15)'  
    }
    
    fig = go.Figure()
    shapes = []
    
    # Sicherstellen, dass die Daten für min/max numerisch sind
    df_numeric = df[['Low', 'High', 'Close']].apply(pd.to_numeric, errors='coerce')
    
    try:
        y_min = df_numeric['Low'].min() * 0.99
        y_max = df_numeric['High'].max() * 1.01
    except:
        y_min = df_numeric['Close'].min() * 0.95
        y_max = df_numeric['Close'].max() * 1.05


    # Füge Rechtecke für jeden Tag mit dem entsprechenden Regime hinzu
    for i in range(len(df)):
        date_start = df.index[i]
        
        if i < len(df) - 1:
            date_end = df.index[i+1]
        else:
            date_end = date_start + pd.Timedelta(days=1) 
            
        regime = df['Regime'].iloc[i]
        color = REGIME_COLORS.get(regime, 'rgba(108, 117, 125, 0.1)') 
        
        shapes.append(
            dict(
                type="rect",
                x0=date_start,
                x1=date_end,
                y0=y_min,
                y1=y_max,
                fillcolor=color,
                layer="below",
                line_width=0,
            )
        )
        
    fig.update_layout(shapes=shapes)

    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='BTC/USD Preis',
        increasing_line_color='#28a745', 
        decreasing_line_color='#dc3545'  
    ))

    # Moving Average Traces
    ma_colors = {
        'MA7': 'blue', 
        'MA21': 'purple', 
        'MA50': 'orange', 
        'MA200': 'red'
    }
    
    for ma, color in ma_colors.items():
        if ma in df.columns:
            df[ma] = pd.to_numeric(df[ma], errors='coerce')
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[ma], 
                mode='lines', 
                name=ma, 
                line=dict(color=color, width=1.5)
            ))

    # Layout-Anpassungen
    fig.update_layout(
        title=f'BTC/USD Kursverlauf, MAs & Markt-Regime ({df.index.min().strftime("%Y-%m-%d")} bis {df.index.max().strftime("%Y-%m-%d")})',
        xaxis_title='Datum',
        yaxis_title='Preis (USD)',
        xaxis_rangeslider_visible=False, 
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white"
    )

    fig.update_xaxes(type='date', rangeselector=None) 
    
    return fig


# def create_pca_scatter_plot(df, focus_date_str):
#     """Erstellt einen interaktiven Scatter Plot der PCA-Komponenten (PC1 vs. PC2),
#     eingefärbt nach dem Markt-Regime, und hebt den Fokus-Tag hervor. (Fix für den PCA Plot)"""
    
#     # Sicherstellen, dass PC1 und PC2 numerisch sind
#     df['PC1'] = pd.to_numeric(df['PC1'], errors='coerce')
#     df['PC2'] = pd.to_numeric(df['PC2'], errors='coerce')
    
#     df_plot = df.copy().dropna(subset=['PC1', 'PC2', 'Regime'])

#     REGIME_COLOR_MAP = {
#         'Bear': '#DC3545',      # Rot
#         'Sideways': '#FFC107',  # Gelb
#         'Bull': '#28A745',      # Grün
#     }
    
#     fig = go.Figure()

#     # 1. Historische Punkte nach Regime
#     for regime, color in REGIME_COLOR_MAP.items():
#         df_regime = df_plot[df_plot['Regime'] == regime]
        
#         # Base historical points
#         fig.add_trace(go.Scatter(
#             x=df_regime['PC1'],
#             y=df_regime['PC2'],
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 color=color,
#                 opacity=0.6,
#                 line=dict(width=0.5, color='DarkSlateGrey')
#             ),
#             name=f'Regime: {regime}',
#             text=df_regime.index.strftime('%Y-%m-%d') + '<br>Regime: ' + df_regime['Regime'],
#             hoverinfo='text+x+y'
#         ))

#     # 2. Fokus-Tag hervorheben
#     # Datum in das gleiche Format konvertieren wie im Index, um den Vergleich zu ermöglichen
#     focus_date_dt = pd.to_datetime(focus_date_str).strftime('%Y-%m-%d')
    
#     # Filtern nach dem Fokus-Tag im formatierten Index
#     focus_data_match = df_plot[df_plot.index.strftime('%Y-%m-%d') == focus_date_dt]
    
#     if not focus_data_match.empty:
#         focus_data = focus_data_match.iloc[0]
        
#         fig.add_trace(go.Scatter(
#             x=[focus_data['PC1']],
#             y=[focus_data['PC2']],
#             mode='markers+text',
#             marker=dict(
#                 size=20,
#                 color='#FBBF24', # Fokus-Farbe (Amber)
#                 line=dict(width=3, color='black'),
#                 symbol='star' # Stern-Symbol für den Fokus
#             ),
#             name='Fokus-Tag',
#             text=[f'🎯 {focus_date_str} ({focus_data["Regime"]})'],
#             textposition="top center",
#             textfont=dict(size=14, color='black', weight='bold'),
#             hoverinfo='text'
#         ))


#     # 3. Layout-Anpassungen
#     fig.update_layout(
#         title='Markt-Regime-Visualisierung (PC1 vs. PC2)',
#         xaxis_title='PC1 (Trend-Achse: Bear ⬅️ ➡️ Bull)',
#         yaxis_title='PC2 (Volatilitäts-Achse)',
#         legend_title="Markt-Regime",
#         height=650,
#         template="plotly_white"
#     )
    
#     # 4. Optional: Markierung des PC1-Durchschnitts zur Trennung von Bull/Bear
#     fig.add_vline(x=df_plot['PC1'].mean(), line_width=1, line_dash="dash", line_color="gray", 
#                   annotation_text="Historischer Durchschnitt PC1", annotation_position="bottom right")

#     return fig

def create_pca_scatter_plot(df, focus_date_str):
    """Erstellt einen interaktiven Scatter Plot der PCA-Komponenten (PC1 vs. PC2),
    eingefärbt nach dem Markt-Regime, und hebt den Fokus-Tag hervor. (Fix für den PCA Plot)"""
    
    # Sicherstellen, dass PC1 und PC2 numerisch sind
    df['PC1'] = pd.to_numeric(df['PC1'], errors='coerce')
    df['PC2'] = pd.to_numeric(df['PC2'], errors='coerce')
    
    df_plot = df.copy().dropna(subset=['PC1', 'PC2', 'Regime'])

    REGIME_COLOR_MAP = {
        'Bear': '#DC3545',      # Rot
        'Sideways': '#FFC107',  # Gelb
        'Bull': '#28A745',      # Grün
    }
    
    fig = go.Figure()

    # 1. Historische Punkte nach Regime
    for regime, color in REGIME_COLOR_MAP.items():
        df_regime = df_plot[df_plot['Regime'] == regime]
        
        # Base historical points
        fig.add_trace(go.Scatter(
            x=df_regime['PC1'],
            y=df_regime['PC2'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                opacity=0.6,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=f'Regime: {regime}',
            text=df_regime.index.strftime('%Y-%m-%d') + '<br>Regime: ' + df_regime['Regime'],
            hoverinfo='text+x+y'
        ))

    # 2. Fokus-Tag hervorheben
    focus_date_dt = pd.to_datetime(focus_date_str).strftime('%Y-%m-%d')
    focus_data_match = df_plot[df_plot.index.strftime('%Y-%m-%d') == focus_date_dt]
    
    if not focus_data_match.empty:
        focus_data = focus_data_match.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[focus_data['PC1']],
            y=[focus_data['PC2']],
            mode='markers+text',
            marker=dict(
                size=20,
                color='#FBBF24', # Fokus-Farbe (Amber)
                line=dict(width=3, color='black'),
                symbol='star' # Stern-Symbol für den Fokus
            ),
            name='Fokus-Tag',
            # Behält den Dart-Emoji im Text bei
            text=[f'🎯 {focus_date_str} ({focus_data["Regime"]})'],
            textposition="top center",
            # HIER: Farbe auf WEISS und Größe auf 16 erhöht
            textfont=dict(size=16, color='white', weight='bold'), 
            hoverinfo='text'
        ))


    # 3. Layout-Anpassungen
    fig.update_layout(
        title='Markt-Regime-Visualisierung (PC1 vs. PC2)',
        xaxis_title='PC1 (Trend-Achse: Bear ⬅️ ➡️ Bull)',
        yaxis_title='PC2 (Volatilitäts-Achse)',
        legend_title="Markt-Regime",
        height=650,
        template="plotly_white"
    )
    
    # 4. Optional: Markierung des PC1-Durchschnitts zur Trennung von Bull/Bear
    fig.add_vline(x=df_plot['PC1'].mean(), line_width=1, line_dash="dash", line_color="gray", 
                  annotation_text="Historischer Durchschnitt PC1", annotation_position="bottom right")

    return fig


def find_analogies(df_master, focus_date, top_k):
    """
    Findet die Top-K ähnlichsten Tage basierend auf dem euklidischen Abstand
    der PCA-Komponenten (PC1, PC2).
    """
    if focus_date not in df_master.index:
        return pd.DataFrame(), None
    
    # 1. Definiere den Fokuspunkt (PCA-Werte)
    # Sicherstellen, dass die Werte als Float interpretiert werden
    df_master['PC1'] = pd.to_numeric(df_master['PC1'], errors='coerce')
    df_master['PC2'] = pd.to_numeric(df_master['PC2'], errors='coerce')
    
    focus_point = df_master.loc[focus_date, ['PC1', 'PC2']].values.astype(np.float64)
    
    # 2. Filtere den Fokus-Tag aus der Historie
    df_history = df_master.drop(focus_date, errors='ignore').copy()
    
    # 3. Berechne den euklidischen Abstand
    history_points = df_history[['PC1', 'PC2']].values.astype(np.float64)
    
    # Berechnung des euklidischen Abstands
    distances = np.linalg.norm(history_points - focus_point, axis=1)
    df_history['Distance'] = distances
    
    # 4. Sortiere und wähle die Top-K Tage aus
    df_analogies = df_history.sort_values(by='Distance', ascending=True).head(top_k)
    
    # 5. Berechne die Outcomes für die Analogien (7-Tage-Return)
    close_series = df_master['Close'].apply(pd.to_numeric, errors='coerce')
    real_returns = []
    
    for index in df_analogies.index:
        try:
            loc_index = df_master.index.get_loc(index)
            future_index = loc_index + 7 # 7 Tage später
            
            if future_index < len(df_master):
                start_price = close_series.loc[index]
                # Verwende .iloc, um den Preis am zukünftigen Index zu erhalten
                end_price = close_series.iloc[future_index] 
                real_return = (end_price / start_price) - 1
            else:
                real_return = np.nan 
        except:
             real_return = np.nan
        
        real_returns.append(real_return)
    
    df_analogies['Real_Return_7d'] = real_returns
    
    # 6. Berechne die zusammenfassenden Metriken
    df_analogies_clean = df_analogies.dropna(subset=['Real_Return_7d'])
    up_count = (df_analogies_clean['Real_Return_7d'] > 0).sum()
    total_count = df_analogies_clean['Real_Return_7d'].count()
    
    summary_metrics = {
        'Tage analysiert': total_count,
        'Durchschn. 7d Return (Real)': df_analogies_clean['Real_Return_7d'].mean() * 100, # In %
        'Anteil Up (7d)': (up_count / total_count) * 100 if total_count > 0 else 0, # In %
        'Median 7d Return (Real)': df_analogies_clean['Real_Return_7d'].median() * 100 # In %
    }
    
    return df_analogies, summary_metrics

# ==================================================================================================
# 5. DASHBOARD LOGIK UND LAYOUT
# ==================================================================================================

if df_master is not None:
    
    # ----------------------------------------------------------------------------------------------
    # VORBEREITUNG DER MODELLE UND DATEN
    # ----------------------------------------------------------------------------------------------
    
    reg_models = ['ridge', 'elasticnet', 'svr_rbf', 'random_forest', 'lasso', 'linear', 'naive']
    dir_models = ['LogisticRegression', 'SVM', 'RandomForest', 'KNN', 'GradientBoosting']
    cluster_models = ['KMeans', 'MiniBatchKMeans', 'GMM', 'VBGMM', 'Spectral']

    # Setze Standardwerte für Selectboxes
    selected_reg_model = reg_models[0]
    selected_dir_model = dir_models[0]
    selected_cluster_model = cluster_models[0]
    
    # --- 4.1 OBERER CONTAINER (Fokus-Elemente mit Datumsbereich-Slider) ---
    st.title("₿ BTC Markt-Regime & Prognose-Dashboard")
    
    focus_container = st.container()
    
    with focus_container:
        st.markdown("### 🎯 Prognose-Fokus")
        
        # Datums-Vorbereitung für Slider
        date_timestamps = df_master.index.tolist()
        date_dt_list = [ts.to_pydatetime() for ts in date_timestamps]
        
        num_days = len(date_dt_list)
        if num_days > 50:
            default_start_date = date_dt_list[-51] 
        else:
            default_start_date = date_dt_list[0]
            
        default_end_date = date_dt_list[-1] 
        
        col_slider = st.columns([1])[0] 

        with col_slider:
            # DATUMSBEREICH SLIDER
            selected_date_range = st.slider(
                "Wählen Sie einen Datumsbereich (Historie):",
                min_value=date_dt_list[0],  
                max_value=date_dt_list[-1], 
                value=(default_start_date, default_end_date), 
                format="YYYY-MM-DD" 
            )
            start_date, end_date = selected_date_range
            
            fokus_tag = end_date.strftime('%Y-%m-%d')
            st.info(f"Analyse-Zeitraum: **{start_date.strftime('%Y-%m-%d')}** bis **{fokus_tag}** | Fokus-Tag: **{fokus_tag}**")

        st.markdown("---") 
        
    # --- 4.2 Sidebar (Konfiguration & Sekundär-Filter) ---
    
    st.sidebar.header("Konfiguration & Filter")
    
    # --- Historie-Filter ---
    st.sidebar.subheader("Historie-Filter")
    alle_regime = df_master['Regime'].unique().tolist()
    ausgewaehlte_regime = st.sidebar.multiselect("Markt-Regime filtern (Tab 1):", options=alle_regime, default=alle_regime)
    if 'Signal' in df_master.columns:
        alle_signale = df_master['Signal'].unique().tolist()
        ausgewaehlte_signale = st.sidebar.multiselect("ML-Signal filtern (Tab 1):", options=alle_signale, default=alle_signale)
    else:
        ausgewaehlte_signale = []
    
    # --- Ähnlichkeits-Analyse ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ähnlichkeits-Analyse (Analogiebildung)")
    # top_k ist hier definiert und wird in find_analogies verwendet
    top_k = st.sidebar.slider("Anzahl ähnlicher Tage (Top-K):", min_value=1, max_value=100, value=50)

    # --- Modell-Orientierung ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Modell-Einstellungen")
    st.sidebar.markdown("*(Die Modellauswahl befindet sich jetzt in den jeweiligen 'Metriken'-Tabs.)*")
    
    
    # --- 4.3 Haupt-Layout (Tabs) ---
    
    tab1, tab_reg, tab_dir, tab_cluster, tab2_new, tab4_new, tab5_new = st.tabs([
        "Historie & Signale (Filter)", 
        "Regressions-Metriken (Preisprognose)",
        "Klassifikations-Metriken (Richtung/Signal)",
        "Clustering-Metriken (Markt-Regime)", 
        "Signal-Analyse & Analogien",
        "Markt-Visualisierung (PCA/Cluster)", 
        "Explorative Analyse (EDA)"
    ])
    
    # --- Tab 1: Historie & Signale (Filter) ---
    with tab1:
        st.header("1. Historie & Signale im Zeitverlauf")
        
        # 1. Filterung des Master-DataFrames
        df_filtered = df_master[df_master['Regime'].isin(ausgewaehlte_regime)]
        if 'Signal' in df_master.columns:
             df_filtered = df_filtered[df_filtered['Signal'].isin(ausgewaehlte_signale)]
             
        # 2. Kontext-Slicing
        try:
            df_display_context = df_filtered.loc[start_date:end_date]
        except Exception as e:
            st.error(f"Fehler beim Filtern des Datumsbereichs: {e}. Zeige die letzten 100 Tage an.")
            df_display_context = df_filtered.tail(100)
            
        # -----------------------------------------------------------
        # CANDLESTICK CHART
        # -----------------------------------------------------------
        st.subheader("1.0 BTC/USD Candlestick Chart mit Markt-Regime-Hintergrund")
        
        if not df_display_context.empty:
            chart_fig = create_candlestick_chart(df_display_context)
            st.plotly_chart(chart_fig, use_container_width=True)
        else:
            st.warning("Keine Daten vorhanden, um den Chart im ausgewählten Zeitraum zu erstellen.")
            
        st.markdown("---") 

        
        # 3. Definition der Spalten-Views
        cols_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'MA7', 'MA21', 'MA50', 'MA200', 'Volatility30', 'Momentum7']
        cols_cluster = ['Cluster_3', 'Regime', 'PC1', 'PC2']
        cols_signals = ['Direction_Pred', 'Prob_Up', 'Signal', 'Pred_1d', 'Pred_7d', 'Pred_30d', 'Pred_90d', 'Pred_365d']

        
        # --- TEIL 1: TECHNISCHE FEATURES (ROHDATEN) ---
        st.subheader("1.1 Technische Features & Preisdaten")
        df_features = df_display_context[[col for col in cols_features if col in df_display_context.columns]]
        df_features_formatted = format_prices(df_features)
        
        st.dataframe(
            df_features_formatted.style.apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), 
            use_container_width=True,
            height=300
        )
        st.markdown(
            "### 📈 Erklärung: Technische Features (Input für die Modelle)\n"
            "➡️ **OHLC:** Open, High, Low, Close definieren die Preisspanne.\n"
            "➡️ **Volume & Return:** Gehandeltes Volumen und tägliche prozentuale Preisänderung.\n"
            "➡️ **MA:** Gleitende Durchschnitte (7/21/50/200) als wichtigste Trendindikatoren.\n"
            "➡️ **Volatilität & Momentum:** Schwankungsbreite (Risiko) und Geschwindigkeit der Preisänderung."
        )
        
        st.markdown("---")
        
        # --- TEIL 2: MARKT-REGIME (CLUSTERING) ---
        st.subheader("1.2 Markt-Regime & PCA-Ergebnisse")
        
        df_cluster = df_display_context[[col for col in cols_cluster if col in df_display_context.columns]]
        df_cluster_formatted = format_cluster_pca(df_cluster)

        st.dataframe(
            df_cluster_formatted.style.apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), 
            use_container_width=True,
            height=300
        )
        st.markdown(
            "### 🛡️ Erklärung: Markt-Regime (Clustering & Dimension)\n"
            "➡️ **Cluster_ID & Regime:** Numerische Kennung und lesbare Bezeichnung der Marktphase (z.B. 'Bull', 'Bear').\n"
            "➡️ **PC1:** Die **wichtigste** statistische Größe, die den **Trend** (von Bär zu Bulle) erfasst.\n"
            "➡️ **PC2:** Die zweitwichtigste Größe, die primär die **Volatilität** und das **'Chaos'** im Markt darstellt."
        )
        
        st.markdown("---")
        
        # --- TEIL 3: MODELL-SIGNALE & PROGNOSEN ---
        st.subheader("1.3 Modell-Signale & Preisprognosen")
        
        df_signals = df_display_context[[col for col in cols_signals if col in df_display_context.columns]]
        df_signals_formatted = format_signals(df_signals)

        st.dataframe(
            df_signals_formatted.style.apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), 
            use_container_width=True,
            height=300
        )
        st.markdown(
            "### 🔮 Erklärung: Modell-Output (Prognosen)\n"
            "➡️ **Direction_Pred / Signal:** Binäre Vorhersage und lesbares Handelssignal für den nächsten Tag (Klassifikation).\n"
            "➡️ **Prob_Up:** Modell-Wahrscheinlichkeit dafür, dass der Preis am nächsten Tag steigt.\n"
            "➡️ **Pred (1d/7d/30d/90d/365d):** Erwartete absolute Preisprognose in US-Dollar für den jeweiligen Zukunftshorizont (Regression)."
        )

        
    # --- Tab 3.1: REGRESSIONS-METRIKEN ---
    with tab_reg:
        st.header("3.1 Regressions-Metriken: Wie hoch ist der Anstieg/Abfall (Preisprognose)?")
        
        if df_reg_metrics is None:
            st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden oder konnte nicht geladen/geparst werden unter `{REGRESSION_METRICS_PATH}`. Bitte Dateiformat prüfen.")
        else:
            col_horiz, col_reg_select = st.columns(2) 
            
            # Hole alle eindeutigen Horizont-Werte für das Selectbox
            unique_horizons = df_reg_metrics['horizon'].unique().tolist()
            default_index = unique_horizons.index('7d') if '7d' in unique_horizons else 0

            with col_horiz:
                prognose_horizont = st.selectbox(
                    "Prognose-Horizont (Ziel):", 
                    options=unique_horizons, 
                    index=default_index,
                    key="reg_horizon_select" 
                )
            
            with col_reg_select:
                selected_reg_model = st.selectbox(
                    "Wählen Sie das zu bewertende Regressionsmodell:",
                    options=reg_models, 
                    index=reg_models.index(selected_reg_model),
                    key="reg_model_tab"
                )
            
            st.subheader("Performance der Modelle auf dem Testset")

            # Filterung funktioniert jetzt dank der Konvertierung in der Lade-Sektion
            df_reg_filtered = df_reg_metrics[df_reg_metrics['horizon'] == prognose_horizont].copy()
            
            if df_reg_filtered.empty:
                st.warning(f"Keine Regressions-Metriken für den Horizont **{prognose_horizont}** vorhanden.")
            else:
                st.markdown(f"**Performance für Horizont: {prognose_horizont}**")
                
                df_display = df_reg_filtered.copy()
                # Umbenennung, jetzt dass 'model' vorhanden sein sollte
                df_display.rename(columns={
                    'model': 'Modell', 'rmse': 'RMSE (Durchschnittlicher Fehler in $)', 
                    'mae': 'MAE (Durchschnittliche Abweichung in $)', 
                    'r2': 'R² (Erklärte Varianz, 1.0 ist perfekt)',
                    'mse': 'MSE' 
                }, inplace=True, errors='ignore')
                
                # Formatierung der Metriken
                df_display['MSE'] = pd.to_numeric(df_display['MSE'], errors='coerce').apply(lambda x: f'{x:,.0f}')
                df_display['RMSE (Durchschnittlicher Fehler in $)'] = pd.to_numeric(df_display['RMSE (Durchschnittlicher Fehler in $)'], errors='coerce').apply(lambda x: f'{x:,.2f}')
                df_display['MAE (Durchschnittliche Abweichung in $)'] = pd.to_numeric(df_display['MAE (Durchschnittliche Abweichung in $)'], errors='coerce').apply(lambda x: f'{x:,.2f}')
                df_display['R² (Erklärte Varianz, 1.0 ist perfekt)'] = pd.to_numeric(df_display['R² (Erklärte Varianz, 1.0 ist perfekt)'], errors='coerce').apply(lambda x: f'{x:.4f}')
                
                # Entferne die 'horizon' Spalte für die Anzeige
                df_display = df_display.drop(columns=['horizon'], errors='ignore')

                st.dataframe(
                    # Anwendung des abgesicherten Stylings
                    df_display.style.apply(highlight_selected_reg_model, axis=1, current_model=selected_reg_model),
                    use_container_width=True,
                    column_order=['Modell', 'RMSE (Durchschnittlicher Fehler in $)', 'MAE (Durchschnittliche Abweichung in $)', 'R² (Erklärte Varianz, 1.0 ist perfekt)', 'MSE']
                )

                best_mae_row = df_reg_filtered.loc[df_reg_filtered['mae'].idxmin()]
                best_model = best_mae_row['model']
                best_mae = f"{best_mae_row['mae']:,.0f}"
                best_r2 = f"{best_mae_row['r2']:.2f}"
                
                if prognose_horizont in ['1d', '7d']:
                    quality_text = f"Die Prognosequalität für den kurzen Horizont von **{prognose_horizont}** ist relativ **hoch**."
                elif prognose_horizont in ['30d', '90d']:
                    quality_text = f"Die Prognosequalität für den mittleren Horizont von **{prognose_horizont}** ist **moderat**."
                else:
                    quality_text = f"Die Prognosequalität für den langfristigen Horizont von **{prognose_horizont}** ist **gering**."

                st.markdown("---")
                st.subheader("💡 Performance-Analyse und Interpretation (Fokus: Aktueller Horizont)")
                st.info(
                    f"**Ergebnis für {prognose_horizont}:** {quality_text} Das Modell **{best_model}** liefert aktuell die besten Ergebnisse in der Preisprognose.\n\n"
                    f"➡️ **Mittlere Abweichung (MAE):** Der durchschnittliche absolute Fehler des besten Modells beträgt **{best_mae} $**.\n"
                    f"➡️ **Erklärte Varianz (R²):** Das Modell **{best_model}** erklärt **{best_r2}** der beobachteten Preisbewegungen."
                )
            
            st.markdown("---")
            st.subheader("Definition der Kennzahlen")
            st.info(
                "**RMSE** & **MAE** (Fehler in $): Der mittlere Vorhersagefehler. **Niedrige Werte sind besser**.\n\n"
                "**MSE** (Mean Squared Error): Quadratischer Fehler. **Niedrige Werte sind besser**.\n\n"
                "**R²** (Erklärte Varianz): Anteil der Varianz, der durch das Modell erklärt wird. **Werte nahe 1.0 sind ideal**."
            )
            
            st.markdown("--")
                            # NEU EINGEFÜGTE ERKLÄRUNG FÜR LAIEN:
            st.subheader("2.1 Erklärungen & Interpretation")
            st.markdown("""
                ## 💡 Erklärungen und Benchmarks: Regressions-Metriken (Preisprognose)

                ### 📊 Fehler-Metriken (RMSE & MAE)
                Diese Werte messen, wie weit unser Modell in **US-Dollar ($)** mit der Preisvorhersage **daneben** liegt. Niedrigere Werte sind **immer** besser!

                * **MAE (Mean Absolute Error):** Der **durchschnittliche absolute Fehler**. Zeigt den tatsächlichen, durchschnittlichen Irrtum in $.
                * **RMSE (Root Mean Squared Error):** Ähnlich wie MAE, aber **große Fehler** (Ausreißer) werden **stärker gewichtet und bestraft** (quadriert).

                #### Was ist "Niedrig" bei BTC? (Kontextabhängig)
                Da der Bitcoin-Preis stark schwankt, beurteilen wir den Fehler **relativ** zum Preisniveau:
                * ✅ **Sehr gut:** Der Fehler (MAE/RMSE) beträgt **weniger als 1%** des aktuellen Bitcoin-Preises.
                * 🟡 **Akzeptabel:** Der Fehler liegt zwischen 1% und 3%.

                ---

                ### 📈 Güte-Metrik (R²)
                Der $\mathbf{R^2}$ (R-Quadrat oder **Erklärte Varianz**) sagt uns, **wie gut** unser Modell die historischen Preisbewegungen **erfasst und erklären** kann.

                * **R² nahe 1.0:** Das Modell erklärt fast die gesamte Schwankung der Preise – **Idealfall**.
                * **R² nahe 0.0:** Das Modell ist nicht besser, als einfach den Durchschnittspreis zu raten.

                #### Was ist "Gut" bei BTC-Prognosen?
                Für komplexe, langfristige Finanzprognosen ist ein hoher $R^2$ schwer zu erreichen:
                * ⭐ **Sehr gut:** $\mathbf{R^2 > 0.7}$
                * 👍 **Akzeptabel:** $\mathbf{R^2 > 0.5}$

                ---

                ## 🔎 **Beispiel-Analyse: Ridge (1 Tag Horizont)**

                Angenommen, Sie haben das Modell **Ridge** und den Horizont **1 Tag** ausgewählt, und die Tabelle zeigt folgende fiktive Ergebnisse (bei einem aktuellen BTC-Preis von $65.000$):

                | Modell | RMSE (Durchschn. Fehler in $) | MAE (Durchschn. Abweichung in $) | R² (Erklärte Varianz) |
                |---|---|---|---|
                | **Ridge** | **$450.00** | **$320.00** | **0.8875** |

                ### Interpretation dieser Beispiel-Zahlen:

                1.  **R² (Güte):** Mit $\mathbf{0.8875}$ ist die erklärte Varianz **extrem hoch** (nahe 1.0).
                    > **Fazit:** ⭐ Das Ridge-Modell fängt fast $\mathbf{89\%}$ der Preisschwankungen am nächsten Tag korrekt ein. Dies deutet auf eine **ausgezeichnete Prognosegüte** hin.

                2.  **MAE (Durchschnittlicher Fehler):** Der durchschnittliche Fehler liegt bei **$320.00$**.
                    > **Fazit:** Bei einem Preis von $65.000$ entspricht dies einem Fehler von nur $\approx 0.49\%$. Dies liegt **weit unter der 1%-Benchmark**. ✅ Das Modell liegt im Durchschnitt sehr nah am tatsächlichen Preis.

                3.  **RMSE (Gewichteter Fehler):** Der RMSE von **$450.00$** ist höher als der MAE.
                    > **Fazit:** Die Differenz zwischen RMSE ($450$) und MAE ($320$) ist relativ groß. Das bedeutet, dass es gelegentlich **große Ausreißer** (fehlerhafte Prognosen) gibt, die den RMSE stärker nach oben ziehen.
                                            """, 
                        unsafe_allow_html=True # Notwendig für die Markdown-Formatierung
                    )
                # ENDE DER NEUEN ERKLÄRUNG

    # # --- Tab 3.2: KLASSIFIKATIONS-METRIKEN ---
    # with tab_dir:
    #     st.header("3.2 Klassifikations-Metriken: Steigt oder fällt der Preis **(Direction/Signal für den nächsten Tag)**?")
        
    #     col_dir_select, col_empty = st.columns([1, 4])
    #     with col_dir_select:
    #         selected_dir_model = st.selectbox(
    #             "Wählen Sie das zu bewertende Klassifikationsmodell:",
    #             options=dir_models, 
    #             index=dir_models.index(selected_dir_model),
    #             key="dir_model_tab"
    #         )
            
    #     st.subheader("Performance der Modelle auf dem Testset")
        
    #     if df_dir_metrics is None:
    #         st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden oder konnte nicht geladen werden unter `{DIRECTION_METRICS_PATH}`.")
    #     else:
    #         st.success("✅ Klassifikations-Metrik-Datei gefunden und geladen.")
            
    #         df_dir_display = df_dir_metrics.copy()
            
    #         # Stelle sicher, dass die Spalte 'Model' oder 'Modell' existiert
    #         if 'Model' in df_dir_display.columns:
    #             df_dir_display.rename(columns={'Model': 'Modell'}, inplace=True, errors='ignore')
    #         elif 'model' in df_dir_display.columns:
    #              df_dir_display.rename(columns={'model': 'Modell'}, inplace=True, errors='ignore')
            
    #         df_dir_display.rename(columns={
    #             'Accuracy': 'Accuracy (Gesamttreffer)', 
    #             'Precision': 'Precision (Zuverlässigkeit)', 'Recall': 'Recall (Erkennung)', 
    #             'F1': 'F1-Score (Balance)' 
    #         }, inplace=True, errors='ignore')

    #         # Korrektur: Konvertierung in Prozent für bessere Lesbarkeit
    #         for col_key in ['Accuracy (Gesamttreffer)', 'Precision (Zuverlässigkeit)', 'Recall (Erkennung)', 'F1-Score (Balance)']: 
    #             if col_key in df_dir_display.columns:
    #                 df_dir_display[col_key] = pd.to_numeric(df_dir_display[col_key], errors='coerce')
    #                 df_dir_display[col_key] = df_dir_display[col_key].apply(lambda x: f'{x*100:.2f} %') 
            
    #         st.dataframe(
    #             # Anwendung des abgesicherten Stylings
    #             df_dir_display.style.apply(highlight_selected_dir_model, axis=1, current_model=selected_dir_model),
    #             use_container_width=True
    #         )
            
    #         # Wähle die beste Zeile basierend auf F1-Score
    #         if 'F1' in df_dir_metrics.columns:
    #             best_f1_row = df_dir_metrics.loc[df_dir_metrics['F1'].idxmax()]
                
    #             # Stelle sicher, dass 'Modell' in der besten Zeile verfügbar ist
    #             best_model_dir = next((best_f1_row[col] for col in ['Model', 'model'] if col in best_f1_row), "N/A")

    #             best_f1 = f"{best_f1_row['F1'] * 100:.2f}%"
    #             best_recall = f"{best_f1_row['Recall'] * 100:.2f}%"
    #             best_precision = f"{best_f1_row['Precision'] * 100:.2f}%"
                
    #             st.markdown("---")
    #             st.subheader("💡 Performance-Analyse und Interpretation (Fokus: Direction-Signal 1d)")
    #             st.info(
    #                 f"**Ergebnis:** Das Modell **{best_model_dir}** bietet mit einem F1-Score von **{best_f1}** die beste Gesamtbalance.\n\n"
    #                 f"➡️ **Erkennung (Recall):** Das Modell identifiziert **{best_recall}** aller tatsächlichen Preisanstiege (minimiert verpasste Kaufgelegenheiten).\n"
    #                 f"➡️ **Zuverlässigkeit (Precision):** Nur in **{best_precision}** der Fälle liegt das Signal richtig, wenn es einen Anstieg vorhersagt (Fehlalarmrate beachten)."
    #             )
    #         else:
    #             st.info("💡 **Analyse-Status:** Die Performance-Interpretation ist nicht verfügbar, da die Spalte 'F1' fehlt.")
            
    #         st.markdown("---")
    #         st.subheader("Definition der Kennzahlen")
    #         st.info(
    #             "**Accuracy**: Gesamtanteil korrekter Vorhersagen. **Werte nahe 100% sind ideal**.\n\n"
    #             "**Precision**: Zuverlässigkeit des Kaufsignals (Wie oft liegt das Signal richtig?).\n\n"
    #             "**Recall**: Vollständigkeit der Erkennung (Wie viele tatsächliche Anstiege wurden erkannt?).\n\n"
    #             "**F1-Score**: Bester Indikator für die Gesamtperformance (Balance zwischen Precision und Recall)."
    #         )

# --- Tab 3.2: KLASSIFIKATIONS-METRIKEN ---
with tab_dir:
    st.header("3.2 Klassifikations-Metriken: Steigt oder fällt der Preis **(Direction/Signal für den nächsten Tag)**?")
    
    col_dir_select, col_empty = st.columns([1, 4])
    with col_dir_select:
        selected_dir_model = st.selectbox(
            "Wählen Sie das zu bewertende Klassifikationsmodell:",
            options=dir_models, 
            index=dir_models.index(selected_dir_model),
            key="dir_model_tab"
        )
        
    st.subheader("Performance der Modelle auf dem Testset")
    
    if df_dir_metrics is None:
        st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden oder konnte nicht geladen werden unter `{DIRECTION_METRICS_PATH}`.")
    else:
        st.success("✅ Klassifikations-Metrik-Datei gefunden und geladen.")
        
        # 🚨 KRITISCHE KORREKTUR: Spaltennamen auf Großschreibung anpassen (behebt KeyError: 'Recall')
        # Wir stellen sicher, dass die Namen, die der Analysecode erwartet, existieren.
        rename_map_analysis = {
            'precision': 'Precision',
            'recall': 'Recall', # <--- FIX FÜR DEN FEHLER
            'f1': 'F1',
            'f1-score': 'F1',
            'f1_score': 'F1',
            'accuracy': 'Accuracy'
        }
        
        # Wende die Umbenennung auf den Haupt-DataFrame an, um den Analyseblock zu reparieren
        df_dir_metrics.rename(columns=rename_map_analysis, inplace=True, errors='ignore')
        
        # --- Start des Display-Codes ---
        df_dir_display = df_dir_metrics.copy()
        
        # Stelle sicher, dass die Spalte 'Model' oder 'Modell' existiert
        if 'Model' in df_dir_display.columns:
            df_dir_display.rename(columns={'Model': 'Modell'}, inplace=True, errors='ignore')
        elif 'model' in df_dir_display.columns:
             df_dir_display.rename(columns={'model': 'Modell'}, inplace=True, errors='ignore')
        
        # Die Umbenennung für die Anzeige kann jetzt die Großbuchstaben verwenden
        df_dir_display.rename(columns={
            'Accuracy': 'Accuracy (Gesamttreffer)', 
            'Precision': 'Precision (Zuverlässigkeit)', 'Recall': 'Recall (Erkennung)', 
            'F1': 'F1-Score (Balance)' 
        }, inplace=True, errors='ignore')

        # Korrektur: Konvertierung in Prozent für bessere Lesbarkeit
        for col_key in ['Accuracy (Gesamttreffer)', 'Precision (Zuverlässigkeit)', 'Recall (Erkennung)', 'F1-Score (Balance)']: 
            if col_key in df_dir_display.columns:
                df_dir_display[col_key] = pd.to_numeric(df_dir_display[col_key], errors='coerce')
                df_dir_display[col_key] = df_dir_display[col_key].apply(lambda x: f'{x*100:.2f} %') 
        
        st.dataframe(
            # Anwendung des abgesicherten Stylings
            df_dir_display.style.apply(highlight_selected_dir_model, axis=1, current_model=selected_dir_model),
            use_container_width=True
        )
        
        # Wähle die beste Zeile basierend auf F1-Score
        if 'F1' in df_dir_metrics.columns: # F1 ist jetzt durch die Korrektur gesichert
            best_f1_row = df_dir_metrics.loc[df_dir_metrics['F1'].idxmax()]
            
            # Stelle sicher, dass 'Modell' in der besten Zeile verfügbar ist
            # Da wir 'Modell' in df_dir_metrics nicht umbenannt haben, müssen wir hier auf die Originalspalte prüfen
            best_model_dir = next((best_f1_row[col] for col in ['Modell', 'Model', 'model'] if col in best_f1_row), "N/A")

            # Diese Zeilen funktionieren jetzt, da 'Recall', 'Precision' und 'F1' Großbuchstaben sind
            best_f1 = f"{best_f1_row['F1'] * 100:.2f}%"
            best_recall = f"{best_f1_row['Recall'] * 100:.2f}%"
            best_precision = f"{best_f1_row['Precision'] * 100:.2f}%"
            
            st.markdown("---")
            st.subheader("💡 Performance-Analyse und Interpretation (Fokus: Direction-Signal 1d)")
            st.info(
                f"**Ergebnis:** Das Modell **{best_model_dir}** bietet mit einem F1-Score von **{best_f1}** die beste Gesamtbalance.\n\n"
                f"➡️ **Erkennung (Recall):** Das Modell identifiziert **{best_recall}** aller tatsächlichen Preisanstiege (minimiert verpasste Kaufgelegenheiten).\n"
                f"➡️ **Zuverlässigkeit (Precision):** Nur in **{best_precision}** der Fälle liegt das Signal richtig, wenn es einen Anstieg vorhersagt (Fehlalarmrate beachten)."
            )
        else:
            # Info-Meldung angepasst, falls F1 aus unbekannten Gründen immer noch fehlt
            st.info("💡 **Analyse-Status:** Die Performance-Interpretation ist nicht verfügbar. Überprüfen Sie, ob die Spalte 'F1' in der CSV existiert.")
        
        st.markdown("---")
        st.subheader("Definition der Kennzahlen")
        st.info(
            "**Accuracy** (Genauigkeit): Gesamtanteil korrekter Vorhersagen. **Werte nahe 100% sind ideal**.\n\n"
            
            "**Precision** (Zuverlässigkeit): Zuverlässigkeit des Signals. Wie oft liegt das Modell richtig, wenn es einen Anstieg vorhersagt? "
            "**Beispiel:** Eine Precision von **60%** bedeutet, dass 6 von 10 'Kauf'-Signalen (tatsächlich) zum Erfolg führen. **Hohe Precision** minimiert Fehlsignale (Fehlalarme).\n\n"
            
            "**Recall** (Vollständigkeit/Sensitivität): Vollständigkeit der Erkennung. Wie viele tatsächliche Anstiege wurden vom Modell erkannt? "
            "**Beispiel:** Ein Recall von **85%** bedeutet, dass das Modell 8,5 von 10 echten Preisanstiegen identifiziert. **Hoher Recall** minimiert verpasste Chancen.\n\n"
            
            "**F1-Score**: Bester Indikator für die Gesamtperformance (Harmonisches Mittel von Precision und Recall). Besonders wichtig bei ungleich verteilten Klassen.\n\n"
            
            "**Support** (Vorkommen): Die absolute Anzahl der Instanzen der jeweiligen Klasse (z.B. 'Up' oder 'Down') im Testdatensatz. Dient zur Bewertung der **Gültigkeit** der Metriken und zeigt die **Klassen-Imbalance** auf."
        )
    # --- Tab 3.3: CLUSTERING-METRIKEN ---
    with tab_cluster:
        st.header("3.3 Clustering-Metriken: In welchem Trend/Markt-Regime bewegen wir uns **(Gesamte Historie)**?")
        
        col_cluster_select, col_empty = st.columns([1, 4])
        with col_cluster_select:
            selected_cluster_model = st.selectbox(
                "Wählen Sie das zu bewertende Clustering-Modell:",
                options=cluster_models, 
                index=cluster_models.index(selected_cluster_model),
                key="cluster_model_tab"
            )
            
        st.subheader("Clustering-Metriken (Interne Validierung)")
        
        if df_cluster_metrics is None:
             st.warning(f"⚠️ WARNUNG: Metrik-Datei nicht gefunden unter `{CLUSTERING_METRICS_PATH}`.")
        else:
            st.success("✅ Clustering-Metrik-Datei gefunden und geladen.")
            
            df_cluster_metrics.rename(columns={
                'model': 'Modell', 'Model': 'Modell',
                'Silhouette': 'Silhouette Score', 'Silhouette_Score': 'Silhouette Score', 'silhouette_score': 'Silhouette Score',
                'DaviesBouldin': 'Davies-Bouldin Index', 'Davies_Bouldin_Index': 'Davies-Bouldin Index', 'davies_bouldin_index': 'Davies-Bouldin Index'
            }, inplace=True, errors='ignore') 
            
            required_cols = ['Modell', 'Silhouette Score', 'Davies-Bouldin Index']
            if all(col in df_cluster_metrics.columns for col in required_cols):
                
                st.markdown("**Performance der Clustering-Modelle**")
                
                # Formatierung für die Anzeige
                df_cluster_metrics['Silhouette Score'] = pd.to_numeric(df_cluster_metrics['Silhouette Score'], errors='coerce').apply(lambda x: f'{x:.3f}')
                df_cluster_metrics['Davies-Bouldin Index'] = pd.to_numeric(df_cluster_metrics['Davies-Bouldin Index'], errors='coerce').apply(lambda x: f'{x:.3f}')

                st.dataframe(
                    # Anwendung des abgesicherten Stylings
                    df_cluster_metrics.style.apply(highlight_selected_cluster_model, axis=1, current_model=selected_cluster_model),
                    use_container_width=True
                )
                
                # Hier nehmen wir an, dass die Spalten numerisch sind, um den besten Index zu finden
                df_cluster_metrics['Silhouette Score Num'] = pd.to_numeric(df_cluster_metrics['Silhouette Score'].str.replace(',', '.'), errors='coerce')
                
                best_cluster_row = df_cluster_metrics.loc[df_cluster_metrics['Silhouette Score Num'].idxmax()]
                best_cluster_model = best_cluster_row['Modell']
                best_silhouette = best_cluster_row['Silhouette Score']
                best_db_index = best_cluster_row['Davies-Bouldin Index']
    
                st.markdown("---")
                st.subheader("💡 Performance-Analyse und Interpretation (Fokus: Markt-Regime-Bildung)")
                st.info(
                    f"**Ergebnis:** Das Modell **{best_cluster_model}** liefert mit einem Silhouette Score von **{best_silhouette}** die klarste und robusteste Unterscheidung der Markt-Regime.\n\n"
                    f"➡️ **Silhouette Score ({best_silhouette}):** **Werte nahe 1.0 sind ideal** (beste Cluster-Qualität).\n"
                    f"➡️ **Davies-Bouldin Index ({best_db_index}):** **Niedrigere Werte (nahe 0.0) sind besser** (beste Cluster-Trennung)."
                )
            else:
                 st.info("💡 **Analyse-Status:** Die Performance-Interpretation ist nicht verfügbar, da wichtige Metrik-Spalten fehlen. Bitte prüfen Sie die Spaltennamen in der geladenen Metrik-Datei.")

            # st.markdown("---")
            # st.subheader("Definition der Kennzahlen")
            # st.info(
            #     "**Silhouette Score**: Misst die Kompaktheit und Trennung der Cluster. **Werte nahe 1.0 sind ideal**.\n"
            #     "**Davies-Bouldin Index**: Misst die Ähnlichkeit zwischen den Clustern. **Werte nahe 0.0 sind ideal**."
            # )
            
            st.markdown("---")
            st.subheader("Definition der Kennzahlen")
            st.info(
                "Die Güte von Clustering-Ergebnissen wird durch **interne Validierungs-Indizes** bewertet, da keine wahren Labels (Ground Truth) existieren. Diese Indizes messen, wie **kompakt** die einzelnen Cluster sind (Homogenität) und wie **klar voneinander getrennt** die Cluster sind (Separierbarkeit)."
            )
            
            st.markdown("---")
            
            ### 📏 Silhouette Score
            st.markdown("#### 📏 Silhouette Score (Kompaktheit & Trennung)")
            st.markdown(
                "Der **Silhouette Score** misst, wie ähnlich ein Objekt seinem eigenen Cluster ist (Kompaktheit) im Vergleich zu anderen Clustern (Trennung). Er liegt zwischen **-1 und +1**.\n\n"
                "➡️ **Werte nahe +1.0** sind **ideal** und bedeuten, dass das Objekt gut zu seinem Cluster passt und von den Nachbarclustern gut getrennt ist.\n"
                "➡️ **Werte nahe 0** deuten auf überlappende Cluster hin.\n"
                "➡️ **Negative Werte** zeigen an, dass ein Objekt dem falschen Cluster zugewiesen wurde."
            )
            
            ### ⚖️ Calinski-Harabasz Index
            st.markdown("#### ⚖️ Calinski-Harabasz Index (Varianz-Verhältnis)")
            st.markdown(
                "Der **Calinski-Harabasz Index** (auch **Varianz-Verhältnis-Kriterium**) bewertet das Verhältnis von **Zwischen-Cluster-Varianz** (B) zu **Innerhalb-Cluster-Varianz** (W).\n\n"
                "$$CH = \\frac{\\text{Varianz zwischen Clustern (B)}}{\\text{Varianz innerhalb der Cluster (W)}}$$\n\n"
                "➡️ **Ein höherer Wert ist ideal**, da er eine klare Trennung der Cluster (hohes B) und gleichzeitig kompakte Cluster (kleines W) signalisiert."
            )
            
            ### 🧩 Davies-Bouldin Index
            st.markdown("#### 🧩 Davies-Bouldin Index (Ähnlichkeit der Cluster)")
            st.markdown(
                "Der **Davies-Bouldin Index** (DBI) ist definiert als der Durchschnitt der maximalen Ähnlichkeit zwischen jedem Cluster und seinem ähnlichsten Nachbarcluster. Er basiert auf dem Verhältnis des durchschnittlichen Abstands der Punkte innerhalb eines Clusters zum Abstand zwischen den Cluster-Zentren.\n\n"
                "➡️ **Werte nahe 0.0** sind **ideal** und weisen auf klar getrennte und kompakte Cluster hin. Er ist besonders nützlich, um die interne Kompaktheit zu bewerten."
            )
            
# --- Tab 2: Signal-Analyse & Analogien ---
with tab2_new:
    st.header("2. Signal-Analyse & Analogien")
    
    # Holen Sie sich das Signal des Fokus-Tages
    focus_signal = df_master.loc[fokus_tag, 'Signal'] if fokus_tag in df_master.index and 'Signal' in df_master.columns else "N/A"
    focus_regime = df_master.loc[fokus_tag, 'Regime'] if fokus_tag in df_master.index and 'Regime' in df_master.columns else "N/A"
    
    st.subheader(f"🎯 Fokus-Tag: **{fokus_tag}** (Regime: **{focus_regime}** | Signal: **{focus_signal}**)")
    st.caption(f"Ähnlichkeitsanalyse basiert auf den Top **{top_k}** ähnlichsten Tagen.")

    # -----------------------------------------------------------
    # TEIL 1: ÄHNLICHKEITSANALYSE (ANALOGIEN)
    # -----------------------------------------------------------
    
    focus_date_dt = pd.to_datetime(fokus_tag)
    
    # --- 1. Durchführung der Ähnlichkeitsanalyse ---
    df_analogies, summary_metrics = find_analogies(df_master, focus_date_dt, top_k)
    
    st.markdown("---")
    st.subheader("2.1 Historische Analogien (Was passierte nach ähnlichen Tagen?)")
    
    # ************************************************************
    # HIER BEGINNT DIE ZENTRALE IF-ELSE-ABFRAGE
    # ************************************************************
    if not df_analogies.empty and summary_metrics['Tage analysiert'] > 0:
        
        # --- ZUSAMMENFASSUNG METRIKEN ---
        summary_df = pd.DataFrame([summary_metrics]).T
        summary_df.columns = ['Ergebnis']
        
        summary_df.iloc[0, 0] = f"{int(summary_df.iloc[0, 0])} Tage"
        summary_df.iloc[1, 0] = f"{summary_df.iloc[1, 0]:.2f} %"
        summary_df.iloc[2, 0] = f"{summary_df.iloc[2, 0]:.2f} %"
        summary_df.iloc[3, 0] = f"{summary_df.iloc[3, 0]:.2f} %"

        st.dataframe(summary_df, use_container_width=True)
        
        # --- DYNAMISCHE INTERPRETATION ---
        anteil_up = summary_metrics['Anteil Up (7d)']
        avg_return = summary_metrics['Durchschn. 7d Return (Real)']
        
        st.markdown("---")
        st.subheader("Interpretation der historischen Unterstützung")
        
        if focus_signal == '📈 Up':
            if anteil_up >= 70:
                analogy_interpretation = f"👍 **Starke Bullish-Bestätigung:** In **{anteil_up:.0f}%** der ähnlichen Fälle stieg der Preis. Der durchschnittliche reale Zuwachs betrug **{avg_return:.2f}%**. Die Historie **unterstützt** das 'Up'-Signal stark."
                st.success(analogy_interpretation)
            elif anteil_up <= 30:
                analogy_interpretation = f"👎 **Widerspruch (Fehlalarm-Risiko):** Nur in **{anteil_up:.0f}%** der Fälle stieg der Preis. Der durchschnittliche reale Return ist **{avg_return:.2f}%**. Die Historie **widerlegt** das 'Up'-Signal stark."
                st.error(analogy_interpretation)
            else:
                analogy_interpretation = f"⚠️ **Unklare Unterstützung:** Die Historie ist mit **{anteil_up:.0f}%** Up-Tagen gemischt. **Vorsicht** ist geboten, da die Analogie keine klare Richtung liefert."
                st.warning(analogy_interpretation)

        elif focus_signal == '📉 Down':
            if anteil_up <= 30:
                analogy_interpretation = f"👍 **Starke Bearish-Bestätigung:** Nur in **{anteil_up:.0f}%** der ähnlichen Fälle stieg der Preis (d.h. er fiel in {100-anteil_up:.0f}% der Fälle). Der durchschnittliche reale Return ist **{avg_return:.2f}%** (meist negativ). Die Historie **unterstützt** das 'Down'-Signal stark."
                st.success(analogy_interpretation)
            elif anteil_up >= 70:
                analogy_interpretation = f"👎 **Widerspruch (Hohes Risiko):** In **{anteil_up:.0f}%** der Fälle stieg der Preis, obwohl das Modell 'Down' sagt. Der durchschnittliche reale Return ist **{avg_return:.2f}%** (meist positiv). Die Historie **widerlegt** das 'Down'-Signal stark."
                st.error(analogy_interpretation)
            else:
                analogy_interpretation = f"⚠️ **Unklare Unterstützung:** Die Historie ist mit **{anteil_up:.0f}%** Up-Tagen gemischt. **Vorsicht** ist geboten, da die Analogie keine klare Richtung liefert."
                st.warning(analogy_interpretation)
        else:
             analogy_interpretation = "⚠️ **Kein klares Signal:** Das ML-Modell lieferte kein klares Up/Down-Signal, daher kann die Analogie nur die historische Verteilung zeigen."
             st.warning(analogy_interpretation)

        st.markdown("---")
        st.markdown(f"### 📋 Details der Top {top_k} Analogien")
        
        analogy_cols = ['Regime', 'PC1', 'PC2', 'Distance', 'Close', 'Return', 'Real_Return_7d']
        
        df_analogy_display = df_analogies[[col for col in analogy_cols if col in df_analogies.columns]].copy()
        
        # Formatting for display
        df_analogy_display['PC1'] = df_analogy_display['PC1'].apply(lambda x: f'{x:.3f}')
        df_analogy_display['PC2'] = df_analogy_display['PC2'].apply(lambda x: f'{x:.3f}')
        df_analogy_display['Close'] = df_analogy_display['Close'].apply(lambda x: f'{x:,.2f}')
        df_analogy_display['Return'] = df_analogy_display['Return'].apply(lambda x: f'{x:.4f}')
        # Korrektur: Formatiere den tatsächlichen Return in Prozent
        df_analogy_display['Real_Return_7d'] = df_analogy_display['Real_Return_7d'].apply(lambda x: f'{x*100:.2f} %') 
        df_analogy_display['Distance'] = df_analogy_display['Distance'].apply(lambda x: f'{x:.3f}')
        
        df_analogy_display.rename(columns={
            'Distance': 'Ähnlichkeit (Abstand)',
            'Real_Return_7d': 'Tatsächlicher 7d Return (Real)',
            'Close': 'Schlusskurs',
            'Return': 'Tägl. Return'
        }, inplace=True)
        
        # Wende das Highlight-Styling auf die Spalte an
        st.dataframe(
            df_analogy_display.style.apply(highlight_return_analogy, axis=0).apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), 
            use_container_width=True
        )
        
        # NEUE ERKLÄRUNG DER ZAHLEN UNTER DER TABELLE (ULTIMATIV KORRIGIERT)
        st.markdown("### 📊 Bedeutung der Kennzahlen in der Analogie-Tabelle")
        st.info(
            "Die Analogiebildung basiert auf der **Hauptkomponentenanalyse (PCA)**, welche die Marktstruktur (Indikatoren wie MAs, Momentum und Volatilität) auf zwei Hauptachsen **PC1** und **PC2** verdichtet. "
            "Der gesuchte **euklidische Abstand** ist die direkte, geometrische Distanz zwischen dem **Fokus-Tag** und einem historischen Tag im **PCA-Raum** (der zweidimensionalen Landkarte). **Je niedriger der Abstand, desto ähnlicher sind die Tage.**"
        )

        st.markdown("---")

        # Erklärung des K-Sliders
        st.markdown("#### ⚙️ Der Top-K Slider (Anzahl ähnlicher Tage)")
        st.markdown(
            "Der Slider **Anzahl ähnlicher Tage (Top-K)** bestimmt, wie viele historische Tage mit dem **geringsten Abstand** (der höchsten Ähnlichkeit) zum Fokus-Tag in der Tabelle angezeigt werden.\n\n"
            "➡️ **Niedrige K-Werte (z.B. K=1 bis 10):** Fokussiert sich auf die **stärksten** Analogien. Die Ergebnisse sind präziser, aber sensibler gegenüber Ausreißern.\n"
            "➡️ **Hohe K-Werte (z.B. K=50 bis 100):** Liefert einen **durchschnittlichen** Eindruck des historischen Verhaltens unter *ähnlichen* Marktbedingungen. Glättet Extremwerte, verwässert aber die stärksten Signale."
        )

        st.markdown("---")
                    
        st.markdown("#### Kennzahlen in der Tabelle")

        # 1. Ähnlichkeit (Abstand)
        st.markdown("##### 1. Ähnlichkeit (Abstand)")
        st.markdown(
            "Dies ist der **euklidische Abstand** zwischen dem **Fokus-Tag** und diesem historischen Tag im PCA-Raum (PC1/PC2). Dieser Wert ist die **Basis** für das Ranking.\n"
            "➡️ **Idealer Wert:** **Niedriger** (Nahe 0.0) ist besser, da dies eine stärkere Korrelation der Marktstrukturen signalisiert."
        )
                    
        # 2. PC1 / PC2 (Marktposition) - FINAL KORRIGIERT
        st.markdown("##### 2. PC1 / PC2 (Marktposition)")
        st.markdown(
            "Diese Werte sind die Koordinaten der **Marktposition** an diesem historischen Tag im PCA-Raum.\n"
            "Die Komponenten fassen Indikatoren wie `MAs`, `Momentum` und `Volatilität` zusammen.\n\n"
            "**Aussage am Markt:**\n"
            "* **PC1 (Trend/Struktur):** Repräsentiert die **Bullische Dynamik** (X-Achse). Hohe positive Werte bedeuten einen starken Aufwärtstrend (z.B. Preis über `MA50`/`MA200`).\n"
            "* **PC2 (Volatilität/Chaos):** Repräsentiert die **Schwankungsbreite** (Y-Achse). Hohe Werte bedeuten turbulente, unsichere Märkte (hohe `Volatility30`).\n\n"
            "➡️ **Bedeutung:** Die PC1/PC2-Werte in der Tabelle sollten **sehr nahe** an den Werten des Fokus-Tages liegen. Das bestätigt, dass die Analogie **strukturell** passt. "
        )
                    
        # 3. Tatsächlicher 7d Return (Real)
        st.markdown("##### 3. Tatsächlicher 7d Return (Real)")
        st.markdown(
            "Dies ist die **tatsächliche prozentuale Preisänderung**, die **in den 7 Tagen nach** diesem historischen Tag (der Analogie) eingetreten ist.\n"
            "➡️ **Interpretation:** Die Spalte dient zur Prognose. Wenn die Mehrheit der Top-K-Tage einen **positiven** Return zeigte, deutet dies auf eine historische Wahrscheinlichkeit für einen Preisanstieg in der kommenden Woche hin."
        )

        st.markdown("---") # Visuelle Trennung des Erklärungsblocks
        
        # -----------------------------------------------------------
        # TEIL 2: SIGNAL-ANALYSE (ERKLÄRUNG)
        # -----------------------------------------------------------
        
        st.markdown("---")
        st.subheader("2.2 Erklärung der Logik und Interpretation")
        
        st.markdown(
            "### 🔎 Signal-Analyse: Was bedeutet das?\n"
            "Dieser Tab verbindet die rohe **Prognose** des Machine-Learning-Modells mit der **historischen Erfahrung** (Analogien).\n\n"
            "#### 1. Der ML-Prognose-Fokus (Signal des Fokus-Tages)\n"
            f"Das ML-Modell hat für den **Fokus-Tag ({fokus_tag})** das Signal **{focus_signal}** mit dem Markt-Regime **{focus_regime}** ausgegeben.\n"
            "Dieses Signal basiert auf der 1-Tages-Prognose der Klassifikationsmodelle, die die aktuellen **technischen Kennzahlen** bewerten.\n\n"
            "#### 2. Die Analogie-Validierung (Historischer Vergleich)\n"
            "Die **Analogien** suchen mithilfe des **PCA-Markt-Regime-Raums** (`PC1`, `PC2`) nach historischen Tagen, die dem Fokus-Tag mathematisch am ähnlichsten waren.\n\n"
            "➡️ **Prüfschritt:** Was geschah **historisch** in den 7 Tagen, nachdem der Markt in der gleichen Konstellation wie heute war?\n\n"
            "#### 3. Interpretation der Ergebnisse (Ihre Frage)\n"
            "Die **Metriken** der Analogien dienen als **Konfidenz-Filter** für die ML-Prognose (siehe die Zusammenfassung oben)."
        )
    
    
# # --- Tab 4: MARKT-VISUALISIERUNG (PCA/Cluster) ---
#     with tab4_new:
#         st.header("4. Markt-Visualisierung (PCA/Cluster)")
#         st.markdown("### 🗺️ Die Position des Fokus-Tags im Markt-Regime-Raum")
        
#         # LOKALE ÜBERSCHREIBUNG: Eindeutige Datumsauswahl nur für die PCA
#         st.markdown("---")
        
#         # *WICHTIG:* Wir nutzen das global definierte end_date als Standardwert
#         # und die bereits vorbereitete Liste date_dt_list
#         fokus_datum_pca = st.date_input(
#             "📆 Wählen Sie den FOKUS-TAG für die PCA-Visualisierung:",
#             value=end_date, # Nutzt das Enddatum des globalen Sliders als Standard
#             min_value=date_dt_list[0],
#             max_value=date_dt_list[-1],
#             key='fokus_pca_selector'
#         )
        
#         # Definiere den lokalen Fokus-Tag, der nur hier verwendet wird
#         fokus_tag_pca = fokus_datum_pca.strftime('%Y-%m-%d')

#         st.info(f"Visualisierter Fokus-Tag: **{fokus_tag_pca}**")
#         st.markdown("---")
        
#         # -----------------------------------------------------------------
#         # NEUER ERKLÄRUNGSBLOCK (OPTIMIERT FÜR PRÄSENTATION)
#         # -----------------------------------------------------------------
#         st.subheader("Detaillierte Erläuterung der Marktstruktur-Visualisierung")
#         st.markdown(
#             "Diese Ansicht basiert auf der **Hauptkomponentenanalyse (PCA)**. Die PCA reduziert die Komplexität von über zwanzig technischen Indikatoren auf zwei Hauptachsen. "
#             "Das Ergebnis ist eine **zweidimensionale Landkarte** des Marktes, die es uns ermöglicht, die aktuelle Marktphase (**Stern/Target**) im Kontext der gesamten historischen Bewegung visuell einzuordnen. "
#         )
        
#         st.markdown("---")
        
#         st.subheader("1. Die drei Komponenten der Visualisierung")
#         st.markdown(
#             "#### a) Datenbasis (Die Historischen Punkte)\n"
#             "**Jeder einzelne Punkt** auf der Grafik repräsentiert die **Marktstruktur eines Handelstages** in unserem Datensatz. Diese Punkte bilden die **historische Datenbank** aller jemals aufgetretenen Marktbedingungen."
#         )
#         st.markdown(
#             "#### b) Markt-Regime (Die Farb-Cluster)\n"
#             "Die Punkte sind mithilfe von Clustering-Algorithmen in **farbige Cluster** (Regime) gruppiert. Ein Cluster fasst Tage mit **statistisch ähnlicher technischer Struktur** zusammen. "
#             "**Fazit:** Tage innerhalb desselben Clusters teilen typischerweise ähnliche Verhaltensmuster, was für die Prognose essenziell ist."
#         )
#         st.markdown(
#             f"#### c) Der Fokus-Tag (⭐/🎯 Target)\n"
#             f"Der große, hervorgehobene **Stern** (**⭐** oder **🎯**) zeigt die **exakte Position** des aktuell gewählten **Fokus-Tages ({fokus_tag_pca})** im Markt-Regime-Raum. "
#             "Seine Position im Verhältnis zu den Clustern bestätigt das von unserem Modell zugewiesene Markt-Regime und dient als visueller Startpunkt für die Analogien-Analyse."
#         )
        
#         # -----------------------------------------------------------------
#         # VISUALISIERUNG UND ACHSEN-ERKLÄRUNG
#         # -----------------------------------------------------------------

#         # pca_fig = create_pca_scatter_plot(df_master, fokus_tag) <-- Veraltet
#         pca_fig = create_pca_scatter_plot(df_master, fokus_tag_pca) # <-- Neu: Nutzt den lokalen Tag
#         st.plotly_chart(pca_fig, use_container_width=True)

#         st.markdown("---")
#         st.subheader("2. Interpretation der Achsen (PC1 & PC2)")
#         st.markdown(
#             "➡️ **X-Achse (PC1): Trend-Komponente (Dominante Marktrichtung).** Repräsentiert die primäre Stärke des Trends. "
#             "Bewegung nach **rechts** signalisiert eine Zunahme der **bullischen Dynamik** (starker Aufwärtstrend). Bewegung nach **links** signalisiert eine starke **bearishe** Tendenz (Abwärtstrend).\n"
#             "➡️ **Y-Achse (PC2): Volatilitäts-Komponente (Markt-Unsicherheit).** Repräsentiert die Schwankungsbreite und das Rauschen im Markt. "
#             "Eine **hohe** Position (oben) deutet auf hohe **Volatilität** und Unsicherheit hin. Niedrige Positionen (unten) stehen für ruhige, stabile Marktphasen.\n"
#             f"➡️ **Stern/Target (⭐/🎯):** Seine Position in diesem Koordinatensystem liefert die Grundlage für die Analogie-Suche in **Tab 2**."
#         )
        
# --- Tab 4: MARKT-VISUALISIERUNG (PCA/Cluster) ---

    # --- Tab 4: MARKT-VISUALISIERUNG (PCA/Cluster) ---
with tab4_new:
    st.header("4. Markt-Visualisierung (PCA/Cluster)")
    st.markdown("### 🗺️ Die Position des Fokus-Tags im Markt-Regime-Raum")
    
    # LOKALE ÜBERSCHREIBUNG: Eindeutige Datumsauswahl nur für die PCA
    st.markdown("---")
    
    # *WICHTIG:* Wir nutzen das global definierte end_date als Standardwert
    # und die bereits vorbereitete Liste date_dt_list
    fokus_datum_pca = st.date_input(
        "📆 Wählen Sie den FOKUS-TAG für die PCA-Visualisierung:",
        value=end_date, # Nutzt das Enddatum des globalen Sliders als Standard
        min_value=date_dt_list[0],
        max_value=date_dt_list[-1],
        key='fokus_pca_selector'
    )
    
    # Definiere den lokalen Fokus-Tag, der nur hier verwendet wird
    fokus_tag_pca = fokus_datum_pca.strftime('%Y-%m-%d')

    st.info(f"Visualisierter Fokus-Tag: **{fokus_tag_pca}**")
    st.markdown("---")
    
    # -----------------------------------------------------------------
    # NEUER ERKLÄRUNGSBLOCK (OPTIMIERT FÜR PRÄSENTATION)
    # -----------------------------------------------------------------
    st.subheader("Erläuterung der Marktstruktur-Visualisierung")
    st.markdown(
        "Diese Ansicht basiert auf der **Hauptkomponentenanalyse (PCA)**. Die PCA reduziert die Komplexität von über zwanzig technischen Indikatoren auf zwei Hauptachsen. "
        "Das Ergebnis ist eine **zweidimensionale Landkarte** des Marktes, die es uns ermöglicht, die aktuelle Marktphase (**Stern/Target**) im Kontext der gesamten historischen Bewegung visuell einzuordnen. "
    )
    
    # -----------------------------------------------------------------
    # VISUALISIERUNG UND ACHSEN-ERKLÄRUNG
    # -----------------------------------------------------------------

    pca_fig = create_pca_scatter_plot(df_master, fokus_tag_pca) # Nutzt nun die Funktion mit weißem Text
    st.plotly_chart(pca_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("1. Interpretation der Achsen (PC1 & PC2)")
    st.markdown(
        "➡️ **X-Achse (PC1): Trend-Komponente (Dominante Marktrichtung).** Repräsentiert die primäre Stärke des Trends. "
        "Bewegung nach **rechts** signalisiert eine Zunahme der **bullischen Dynamik** (starker Aufwärtstrend). Bewegung nach **links** signalisiert eine starke **bearishe** Tendenz (Abwärtstrend).\n\n"
        
        "➡️ **Y-Achse (PC2): Volatilitäts-Komponente (Markt-Unsicherheit).** Repräsentiert die Schwankungsbreite und das Rauschen im Markt. "
        "Eine **hohe** Position (oben) deutet auf hohe **Volatilität** und Unsicherheit hin. Niedrige Positionen (unten) stehen für ruhige, stabile Marktphasen.\n\n"
        
        f"➡️ **Stern/Target (⭐/🎯):** Seine Position in diesem Koordinatensystem liefert die Grundlage für die Analogie-Suche in **Tab 2**."
    )
    
    # -----------------------------------------------------------------
    # NEU: BEISPIELANALYSE DER PC-WERTE
    # -----------------------------------------------------------------
    
    st.markdown("--")
    
    st.subheader("2. Interpretation der Koordinaten (PC-Werte)")
    st.markdown(
        "Wenn Sie über einen Punkt (einen Tag) fahren, sehen Sie dessen exakte **Koordinaten** (PC1- und PC2-Werte). Diese Werte sind normiert, liegen also typischerweise zwischen ca. -0.1 und +0.1."
    )
    
    st.markdown("#### 📈 PC1-Werte (Trend-Komponente)")
    st.markdown(
        "* **Hoher PC1-Wert (positiv, z.B. $> 0.05$):** Der Tag weist eine **stark bullische Struktur** auf (hohes Momentum, weit über gleitenden Durchschnitten). Der Markt bewegte sich dominant nach oben.\n"
        "* **Niedriger PC1-Wert (negativ, z.B. $< -0.05$):** Der Tag weist eine **stark bearishe Struktur** auf. Der Markt befand sich in einem deutlichen Abwärtstrend.\n"
        "* **PC1 nahe Null:** Der Tag hatte eine **neutrale** (seitwärts gerichtete) Trendstruktur."
    )
    
    st.markdown("#### 🌪️ PC2-Werte (Volatilitäts-Komponente)")
    st.markdown(
        "* **Hoher PC2-Wert (positiv, z.B. $> 0.04$):** Der Tag hatte eine **sehr hohe Volatilität** (hohe Schwankungsbreite, große Kerzen). Dies deutet oft auf Phasen von **Angst oder Gier** hin.\n"
        "* **Niedriger PC2-Wert (negativ oder nahe Null):** Der Tag war **ruhig und stabil** (niedrige Volatilität). Niedrige Werte bedeuten oft geringes Handelsinteresse oder eine Konsolidierungsphase."
    )
    
    st.markdown("---")
    st.markdown("#### 💡 Konkretes Beispiel")
    st.markdown(
        "Nehmen wir an, der **Fokus-Tag** zeigt die Koordinaten **PC1 = 0.065** und **PC2 = 0.015**:\n"
        "1.  **PC1 (0.065):** Der Wert ist hoch und positiv. Interpretation: **Stark bullischer Trend** am Markt.\n"
        "2.  **PC2 (0.015):** Der Wert ist niedrig (nahe Null). Interpretation: **Niedrige bis moderate Volatilität**.\n"
        "**Gesamt:** Der Tag befand sich in einem **starken, aber relativ ruhigen Aufwärtstrend** (Regime wahrscheinlich 'Bull')."
    )
    
    st.markdown("---")
    
    st.subheader("3. Die drei Komponenten der Visualisierung")
    st.markdown(
        "#### a) Datenbasis (Die Historischen Punkte)\n"
        "**Jeder einzelne Punkt** auf der Grafik repräsentiert die **Marktstruktur eines Handelstages** in unserem Datensatz. Diese Punkte bilden die **historische Datenbank** aller jemals aufgetretenen Marktbedingungen."
    )
    st.markdown(
        "#### b) Markt-Regime (Die Farb-Cluster)\n"
        "Die Punkte sind mithilfe von Clustering-Algorithmen in **farbige Cluster** (Regime) gruppiert. Ein Cluster fasst Tage mit **statistisch ähnlicher technischer Struktur** zusammen. "
        "**Fazit:** Tage innerhalb desselben Clusters teilen typischerweise ähnliche Verhaltensmuster, was für die Prognose essenziell ist."
    )
    st.markdown(
        f"#### c) Der Fokus-Tag (⭐/🎯 Target)\n"
        f"Der große, hervorgehobene **Stern** (**⭐** oder **🎯**) zeigt die **exakte Position** des aktuell gewählten **Fokus-Tages ({fokus_tag_pca})** im Markt-Regime-Raum. "
        "Seine Position im Verhältnis zu den Clustern bestätigt das von unserem Modell zugewiesene Markt-Regime und dient als visueller Startpunkt für die Analogien-Analyse."
    )
    

    with tab5_new:
        st.header("5. Explorative Analyse (EDA)")
        st.info("Dieser Tab ist für zukünftige erweiterte Analysen, z.B. Feature-Wichtigkeiten, gedacht.")
        st.dataframe(df_master.tail(10).style.apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), use_container_width=True)


# ==================================================================================================
# ENDE
# ==================================================================================================