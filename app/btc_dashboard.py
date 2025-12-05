import streamlit as st
import pandas as pd
import os
import joblib 
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.neighbors import KernelDensity


# ==================================================================================================
# 1. STREAMLIT KONFIGURATION
# ==================================================================================================
st.set_page_config(layout="wide", page_title="₿ BTC Markt-Regime & Prognose-Dashboard") 

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

def load_direction_metrics():
    path = "/Users/burcukiran/Desktop/Abschlussprojekt_Data_Science/models/direction/direction_model_metrics.csv"
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("Fehler beim Laden der Direction-Metriken:", e)
        return None
    
def create_direction_model_performance_plot():
    df = load_direction_metrics()
    if df is None:
        return go.Figure().add_annotation(
            text="Direction-Metriken konnten nicht geladen werden.",
            x=0.5, y=0.5, showarrow=False
        )

    # F1-Spalte automatisch finden
    f1_col = None
    for col in df.columns:
        if "f1" in col.lower():
            f1_col = col
            break

    if f1_col is None:
        return go.Figure().add_annotation(
            text=f"Keine F1-Spalte gefunden. Spalten: {list(df.columns)}",
            x=0.5, y=0.5, showarrow=False
        )

    df_grouped = df.groupby("Modell")[f1_col].mean().reset_index()
    df_grouped["f1_pct"] = df_grouped[f1_col] * 100

    fig = px.bar(
        df_grouped,
        x="Modell",
        y="f1_pct",
        color="Modell",
        title="Prüfung These 7: Modellgenauigkeit (F1-Score) für Up/Down-Vorhersage",
        labels={"f1_pct": "F1-Score (%)"},
        text=df_grouped["f1_pct"].round(1).astype(str) + "%",
        template="plotly_white"
    )

    fig.update_yaxes(range=[40, 80])
    fig.update_traces(textposition="outside")

    return fig
# ==================================================================================================
# 3. DATENLADUNG STARTEN
# ==================================================================================================

df_master = load_data(DATA_PATH)


# --- KORREKTUR FÜR CLUSTER_3 DATENTYP (HIER EINFÜGEN) ---
# Konvertiert 0.0, 1.0, 2.0 zu den Strings '0', '1', '2' für Plotly

df_master['Cluster_3'] = df_master['Cluster_3'].fillna(-1).astype(int).astype(str)
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
            text=[f'🎯 {focus_date_str} ({focus_data["Regime"]})'],
            textposition="top center",
            # HIER: Farbe auf WEISS und Größe auf 16 erhöht (Lesbarkeit bei dunklem Hintergrund)
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


# Globale Farbkarte für Konsistenz
REGIME_COLOR_MAP = {
    'Bear': '#DC3545',      # Rot
    'Sideways': '#FFC107',  # Gelb
    'Bull': '#28A745',      # Grün
}

# # --- 1. EDA-FUNKTIONEN (Sektion 5.1) ---

def create_return_distribution_plot(df):
    """Histogramm + Boxplot der täglichen Renditen (EDA)."""
    fig = px.histogram(
        df,
        x="Return",
        nbins=120,
        marginal="box",
        title="Verteilung der täglichen Renditen (Return)",
        template="plotly_white"
    )
    fig.update_xaxes(title="Tägliche Rendite (%)")
    fig.update_yaxes(title="Häufigkeit")
    return fig

def create_correlation_heatmap(df):
    """Korrelationen der wichtigsten Features (EDA)."""
    cols = ["Return", "Volatility30", "Momentum7", "MA50", "MA200"]
    corr = df[cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Korrelationsmatrix der Markt-Features",
        labels=dict(color="Korrelationswert")
    )
    fig.update_layout(height=500)
    return fig

def create_regime_frequency_bar_chart(df):
    """Erstellt ein Balkendiagramm der Häufigkeit der Markt-Regime."""
    
    df_counts = df['Regime'].value_counts().reset_index()
    df_counts.columns = ['Regime', 'Anzahl Tage']

    order = ['Bull', 'Sideways', 'Bear']
    df_counts['Regime'] = pd.Categorical(df_counts['Regime'], categories=order, ordered=True)
    df_counts = df_counts.sort_values('Regime')

    fig = px.bar(
        df_counts,
        x='Regime',
        y='Anzahl Tage',
        color='Regime',
        color_discrete_map=REGIME_COLOR_MAP,
        text='Anzahl Tage',
        title='Häufigkeit der Markt-Regime'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=420, yaxis_title="Anzahl Tage", xaxis_title=None, showlegend=False)

    return fig



# def create_regime_frequency_bar_chart(df):
#     """Erstellt ein Balkendiagramm, das die absoluten Häufigkeiten der Markt-Regime zeigt (Teil 1)."""
#     df_counts = df['Regime'].value_counts().reset_index()
#     df_counts.columns = ['Regime', 'Anzahl Tage']
#     order = ['Bull', 'Sideways', 'Bear']
#     df_counts['Regime'] = pd.Categorical(df_counts['Regime'], categories=order, ordered=True)
#     df_counts = df_counts.sort_values('Regime')
    
#     fig = px.bar(df_counts, x='Regime', y='Anzahl Tage', color='Regime', color_discrete_map=REGIME_COLOR_MAP,
#                  title='Häufigkeit der Markt-Regime', text='Anzahl Tage')
#     fig.update_layout(xaxis_title=None, yaxis_title='Anzahl Tage', showlegend=False, height=400)
#     fig.update_traces(texttemplate='%{text}', textposition='outside')
#     return fig

# def create_cluster_frequency_bar_chart(df):
#     """Erstellt ein Balkendiagramm, das die absoluten Häufigkeiten der ML-Cluster zeigt (Teil 1)."""
#     df_counts = df['Cluster_3'].value_counts().reset_index()
#     df_counts.columns = ['Cluster', 'Anzahl Tage']
#     df_counts['Cluster'] = df_counts['Cluster'].astype(str)
    
#     fig = px.bar(df_counts, x='Cluster', y='Anzahl Tage', color='Cluster',
#                  title='Häufigkeit der ML-Cluster (K=3)', text='Anzahl Tage')
#     fig.update_layout(xaxis_title=None, yaxis_title='Anzahl Tage', showlegend=False, height=400)
#     fig.update_traces(texttemplate='%{text}', textposition='outside')
#     return fig

# def create_time_series_plot(df):
#     """Zeitlicher Verlauf von Preis und Volatilität (Teil 1)."""
#     df['Date'] = pd.to_datetime(df.index)
#     df_ts = df[['Close', 'Volatility30']].copy().dropna()
    
#     fig = go.Figure()
    
#     # Preis (Close) auf primärer Y-Achse
#     fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Close'], name='Close Preis', yaxis='y1', line=dict(color='blue')))
    
#     # Volatilität auf sekundärer Y-Achse
#     fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Volatility30'], name='Volatility30', yaxis='y2', line=dict(color='orange', dash='dot')))
    
#     fig.update_layout(
#         title='Zeitlicher Verlauf von Preis und Volatilität (30d)',
#         xaxis_title='Datum',
#         yaxis=dict(title='Schlusskurs (Close)', color='blue'),
#         yaxis2=dict(title='Volatilität (30d)', overlaying='y', side='right', color='orange', showgrid=False),
#         height=500,
#         legend=dict(x=0.01, y=0.99)
#     )
#     return fig

# ========================================================
# PLOT-FUNKTIONEN FÜR 5.2 THESENPRÜFUNG
# =========================================================

# THESEN BLOCK 1: These 1 (Regime-Unterschiede via Histogramme)
def create_return_by_regime_histograms(df):
    """Erzeugt drei Histogramme der täglichen Renditen – je eines für Bear-, Sideways- und Bull-Phasen (These 1).
    Ziel: Regime sollen leicht verständlich und visuell klar unterscheidbar dargestellt werden.
    """

    import plotly.subplots as sp
    import plotly.graph_objects as go

    if 'Regime' not in df.columns or 'Return' not in df.columns:
        return go.Figure().add_annotation(
            text="Fehlende Spalten (Regime/Return) für These 1.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    regime_order = ['Bear', 'Sideways', 'Bull']
    colors = {'Bear': '#EF553B', 'Sideways': '#FECB52', 'Bull': '#00CC96'}

    # Return numerisch setzen
    df['Return'] = pd.to_numeric(df['Return'], errors='coerce')
    df_plot = df.dropna(subset=['Regime', 'Return'])

    # Subplots erstellen: 1 Zeile, 3 Spalten
    fig = sp.make_subplots(
        rows=1, cols=3,
        subplot_titles=("Bear", "Sideways", "Bull"),
        horizontal_spacing=0.12
    )

    for i, regime in enumerate(regime_order):
        regime_data = df_plot[df_plot['Regime'] == regime]['Return']

        fig.add_trace(
            go.Histogram(
                x=regime_data,
                marker_color=colors[regime],
                name=regime,
                opacity=0.75
            ),
            row=1, col=i+1
        )

        # Achsentitel + Prozentformat
        fig.update_xaxes(title_text="Rendite (%)", tickformat=".1%", row=1, col=i+1)
        fig.update_yaxes(title_text="Häufigkeit", row=1, col=1)

    fig.update_layout(
        title="Prüfung These 1: Histogramme der täglichen Renditen pro Markt-Regime",
        template="plotly_white",
        height=450,
        showlegend=False
    )

    return fig

# THESEN BLOCK 2: These 2 (Cluster-Phasen)
def create_return_by_cluster_violinplot(df):
    """Erzeugt ein Violinplot der täglichen Rendite gruppiert nach ML-Cluster (These 2).
    These 2: Die ML-Cluster (Cluster_3) bilden strukturell unterschiedliche Marktphasen ab,
    die sich in der Renditeverteilung widerspiegeln.
    """

    if 'Cluster_3' not in df.columns or 'Return' not in df.columns:
        return go.Figure().add_annotation(
            text="Fehlende Spalten (Cluster_3/Return) für These 2.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Copy
    df_plot = df.copy()

    # FIX 1: Cluster_3 sicher als numerisch casten
    df_plot['Cluster_3'] = pd.to_numeric(df_plot['Cluster_3'], errors='coerce')

    # FIX 2: Nur gültige Cluster >= 0 behalten
    df_plot = df_plot[df_plot['Cluster_3'].notna() & (df_plot['Cluster_3'] >= 0)]

    # Cluster-ID in String für Kategorien
    df_plot['Cluster_3_str'] = df_plot['Cluster_3'].astype(int).astype(str)

    # Return casten
    df_plot['Return'] = pd.to_numeric(df_plot['Return'], errors='coerce')
    df_plot.dropna(subset=['Cluster_3_str', 'Return'], inplace=True)

    # Sortierte Cluster-Reihenfolge
    cluster_order = sorted(df_plot['Cluster_3_str'].unique(), key=lambda x: int(x))

    # Plot
    fig = px.violin(
        df_plot,
        x='Cluster_3_str',
        y='Return',
        category_orders={'Cluster_3_str': cluster_order},
        color='Cluster_3_str',
        title='Prüfung These 2: Renditeverteilung nach ML-Cluster (Cluster_3)',
        labels={'Return': 'Tägliche Rendite (%)', 'Cluster_3_str': 'ML-Cluster'},
        template='plotly_white',
        box=True,
        points="all"
    )

    fig.update_layout(yaxis_tickformat='.2%')
    fig.update_xaxes(title_text='ML-Cluster ID')
    fig.update_yaxes(title_text='Tägliche Rendite')

    return fig


# THESEN BLOCK 3: These 3 (Hitrate pro Cluster)
def create_hitrate_by_cluster_bar_chart(df):
    """Erzeugt ein Balkendiagramm der Trefferquote pro Cluster (These 3).
    These 3: Das Prognosemodell arbeitet in einigen Clustern besser als in anderen.
    """

    try:
        # ❗ DIESER TEIL IST SIMULIERT – später mit echten Daten ersetzen
        hitrate_data = pd.DataFrame({
            'Cluster_3': [0, 1, 2],
            'Hit_Rate': [0.52, 0.75, 0.65],
            'Cluster_Label': ['Cluster 0', 'Cluster 1', 'Cluster 2']
        })

        hitrate_data = hitrate_data.sort_values(by='Hit_Rate', ascending=False)

        fig = px.bar(
            hitrate_data,
            x='Cluster_Label',
            y='Hit_Rate',
            color='Hit_Rate',
            color_continuous_scale=px.colors.sequential.Teal,
            title='Prüfung These 3: Trefferquote nach ML-Cluster',
            labels={'Hit_Rate': 'Trefferquote', 'Cluster_Label': 'ML-Cluster'},
            template='plotly_white'
        )

        fig.update_yaxes(range=[0.4, 0.9], tickformat='.1%')
        fig.update_xaxes(title_text='ML-Cluster (sortiert)')
        fig.update_yaxes(title_text='Trefferquote (Hit Rate)')

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Datenfehler für These 3: {e}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    return fig


# # THESEN BLOCK 4: These 4 (Cluster vs. Regime Mapping)
# def create_cluster_regime_heatmap(df):
#     """Erstellt eine Heatmap der Häufigkeiten, die zeigt, wie Cluster_3 auf die Regime mappt (These 4).
#     These: Die ML-Cluster korrelieren stark mit den Regime-Labels, sind aber feiner granuliert.
#     """

#     # Sicherstellen, dass Spalten vorhanden sind
#     if 'Regime' not in df.columns or 'Cluster_3' not in df.columns:
#         return go.Figure().add_annotation(
#             text="Daten für These 4 fehlen (Regime oder Cluster_3 Spalte nicht gefunden).",
#             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
#         )

#     # Copy
#     df_plot = df.copy()

#     # FIX 1: Cluster_3 numerisch machen (egal ob string, float, object…)
#     df_plot['Cluster_3'] = pd.to_numeric(df_plot['Cluster_3'], errors='coerce')

#     # FIX 2: Nur gültige Cluster >= 0 (NaN & -1 entfernen)
#     df_plot = df_plot[df_plot['Cluster_3'].notna() & (df_plot['Cluster_3'] >= 0)]

#     # Für Plot zu String casten
#     df_plot['Cluster_3'] = df_plot['Cluster_3'].astype(int).astype(str)

#     # Crosstab erstellen
#     cross_tab = pd.crosstab(df_plot['Regime'], df_plot['Cluster_3'])

#     # Regime-Reihenfolge (falls alle vorhanden sind)
#     regime_order = ['Bear', 'Sideways', 'Bull']
#     if all(r in cross_tab.index for r in regime_order):
#         cross_tab = cross_tab.reindex(regime_order, axis=0)

#     # Heatmap
#     fig = px.imshow(
#         cross_tab,
#         text_auto=True,
#         color_continuous_scale='Viridis',
#         title='Prüfung These 4: Zuordnung der ML-Cluster zu Markt-Regime (Crosstab)',
#         labels={'x': 'ML Cluster-ID (Cluster_3)', 'y': 'Markt-Regime', 'color': 'Anzahl Tage'}
#     )

#     fig.update_xaxes(title="ML Cluster-ID (Cluster_3)")
#     fig.update_yaxes(title="Markt-Regime")

#     # Colorbar verbessern
#     max_count = cross_tab.values.max()
#     fig.update_layout(
#         coloraxis_colorbar=dict(
#             title="Anzahl Tage",
#             tickvals=[0, max_count],
#             ticktext=["Gering", "Hoch"],
#         ),
#         height=450
#     )

#     return fig


# THESEN BLOCK 5: These 5 (Volatilität vs. Rendite)
def create_return_vs_volatility_scatter(df):
    """Erzeugt ein Scatter Plot von Rendite vs. Volatilität mit Korrelationslinie (These 5).
    These: Hohe Volatilität korreliert negativ mit der Rendite (Chaos führt zu Verlust/negativer Schiefe).
    """
    if 'Volatility30' not in df.columns or 'Return' not in df.columns or 'Regime' not in df.columns:
        return go.Figure().add_annotation(text="Fehlende Spalten (Volatility30/Return/Regime) für These 5.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)


    regime_color_map = {'Bear': '#EF553B', 'Sideways': '#FECB52', 'Bull': '#00CC96'}

    # Sicherstellen, dass die Spalten numerisch sind und NaN entfernt werden
    df_plot = df.copy()
    df_plot['Return'] = pd.to_numeric(df_plot['Return'], errors='coerce')
    df_plot['Volatility30'] = pd.to_numeric(df_plot['Volatility30'], errors='coerce')
    df_plot.dropna(subset=['Volatility30', 'Return', 'Regime'], inplace=True)

    fig = px.scatter(
        df_plot,
        x='Volatility30',
        y='Return',
        color='Regime',
        color_discrete_map=regime_color_map,
        title='Prüfung These 5: Rendite vs. Volatilität (30d)',
        labels={'Return': 'Tägliche Rendite (%)', 'Volatility30': 'Volatilität (30 Tage)'},
        template='plotly_white',
        opacity=0.6,
        trendline="ols", # Ordinary Least Squares (OLS) Regression hinzufügen
        trendline_color_override='gray'
    )

    fig.update_layout(yaxis_tickformat='.2%')
    fig.update_xaxes(title_text='Volatilität (30d)')
    fig.update_yaxes(title_text='Tägliche Rendite')

    return fig

# THESEN BLOCK 6: These 6 (MA-Differenz)
def create_ma_diff_by_regime_boxplot(df):
    """Erzeugt ein Boxplot der MA50 - MA200 Differenz nach Markt-Regime (These 6).
    These: Die MA-Differenz (MA50 - MA200) ist ein valider Indikator für das Regime.
    """
    if 'MA50' not in df.columns or 'MA200' not in df.columns or 'Regime' not in df.columns:
        return go.Figure().add_annotation(text="Fehlende Spalten (MA50/MA200/Regime) für These 6.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)


    df_temp = df.copy()

    # Sicherstellen, dass 'MA50' und 'MA200' numerisch sind und die Differenz berechnet wird
    try:
        df_temp['MA50'] = pd.to_numeric(df_temp['MA50'], errors='coerce')
        df_temp['MA200'] = pd.to_numeric(df_temp['MA200'], errors='coerce')
        df_temp.dropna(subset=['MA50', 'MA200', 'Regime'], inplace=True)
        df_temp['MA_Diff'] = df_temp['MA50'] - df_temp['MA200']
    except KeyError as e:
        print(f"Fehler: Spalte {e} fehlt im DataFrame für These 6.")
        return go.Figure().add_annotation(text=f"Daten für These 6 fehlen. {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    regime_order = ['Bear', 'Sideways', 'Bull']
    regime_color_map = {'Bear': '#EF553B', 'Sideways': '#FECB52', 'Bull': '#00CC96'}

    fig = px.box(
        df_temp,
        x='Regime',
        y='MA_Diff',
        category_orders={'Regime': regime_order},
        color='Regime',
        color_discrete_map=regime_color_map,
        title='Prüfung These 6: MA50 - MA200 Differenz nach Markt-Regime (Boxplot)',
        labels={'MA_Diff': 'Differenz MA50 - MA200 (USD)', 'Regime': 'Markt-Regime'},
        template='plotly_white'
    )

    # Horizontale Linie bei y=0
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="gray")
    fig.update_xaxes(title_text='Markt-Regime')
    fig.update_yaxes(title_text='MA-Differenz (USD)')

    return fig



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
        st.subheader("Detaillierte Erläuterung der Marktstruktur-Visualisierung")
        st.markdown(
            "Diese Ansicht basiert auf der **Hauptkomponentenanalyse (PCA)**. Die PCA reduziert die Komplexität von über zwanzig technischen Indikatoren auf zwei Hauptachsen. "
            "Das Ergebnis ist eine **zweidimensionale Landkarte** des Marktes, die es uns ermöglicht, die aktuelle Marktphase (**Stern/Target**) im Kontext der gesamten historischen Bewegung visuell einzuordnen."
        )
        
        st.markdown("---")
        
        st.subheader("1. Die drei Komponenten der Visualisierung")
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
        
        # -----------------------------------------------------------------
        # VISUALISIERUNG UND ACHSEN-ERKLÄRUNG
        # -----------------------------------------------------------------

        pca_fig = create_pca_scatter_plot(df_master, fokus_tag_pca)
        st.plotly_chart(pca_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Interpretation der Achsen (PC1 & PC2)")
        st.markdown(
            "➡️ **X-Achse (PC1): Trend-Komponente (Dominante Marktrichtung).** Repräsentiert die primäre Stärke des Trends. "
            "Bewegung nach **rechts** signalisiert eine Zunahme der **bullischen Dynamik** (starker Aufwärtstrend). Bewegung nach **links** signalisiert eine starke **bearishe** Tendenz (Abwärtstrend).\n\n"
            
            "➡️ **Y-Achse (PC2): Volatilitäts-Komponente (Markt-Unsicherheit).** Repräsentiert die Schwankungsbreite und das Rauschen im Markt. "
            "Eine **hohe** Position (oben) deutet auf hohe **Volatilität** und Unsicherheit hin. Niedrige Positionen (unten) stehen für ruhige, stabile Marktphasen.\n\n"
            
            f"➡️ **Stern/Target (⭐/🎯):** Seine Position in diesem Koordinatensystem liefert die Grundlage für die Analogie-Suche in **Tab 2**."
        )
        
        st.markdown("---")
        
        # -----------------------------------------------------------------
        # BEISPIELANALYSE DER PC-WERTE
        # -----------------------------------------------------------------
        
        st.subheader("3. Interpretation der Koordinaten (PC-Werte)")
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

# --- Tab 5: EXPLORATIVE ANALYSE (EDA) & THESEN ---
    with tab5_new:
        st.header("5. Explorative Analyse (EDA) & Thesen")
        st.info("Dieser Tab beleuchtet die **Datenbasis und Feature-Verteilung** und dient der empirischen **Prüfung der Haupt-Thesen** aus der Marktanalyse.")
        



        # -----------------------------------------------------------------
        # 5.1 DATEN-ÜBERBLICK & FEATURE-EXPLORATION (Die Rohdaten-Basis)
        # -----------------------------------------------------------------
        
        st.subheader("5.1 Daten-Überblick: Häufigkeiten und Verteilungen")
        st.markdown("Der erste Schritt der Analyse ist die Prüfung der Datenbasis und der Verteilung der wichtigsten Markt-Features.")
        
        st.markdown("---")
        
        st.markdown("### 📊 Verteilung der täglichen Renditen")
        st.plotly_chart(create_return_distribution_plot(df_master), use_container_width=True)
        st.caption("Analyse: Die tägliche Rendite ist stark um 0 % zentriert, zeigt aber eine deutlich \
                linksschiefe Verteilung mit vielen Ausreißern. Die meisten Tage liegen im Bereich kleiner \
                Bewegungen (–2 % bis +2 %), während seltene Extremereignisse deutlich weiter ausschlagen. \
                Diese heavy-tailed Struktur erklärt die hohe Marktvolatilität und bildet die Grundlage für \
                die späteren Regime-Unterschiede. → Die Renditen sind nicht normalverteilt, sondern stark \
                schwankungsanfällig."
                )


        st.markdown("### 🔗 Korrelationsmatrix der wichtigsten Features")
        st.plotly_chart(create_correlation_heatmap(df_master), use_container_width=True)
        st.caption("Analyse: Die Korrelationsmatrix zeigt, dass MA50 und MA200 extrem stark miteinander korrelieren \
                (r ≈ 0.98), was typisch für langfristige Trendindikatoren ist. Die Tagesrenditen hingegen weisen \
                nahezu keine Korrelation zu anderen Features auf – ein Hinweis auf die hohe Zufälligkeit von \
                Tagesschwankungen. Volatilität korreliert leicht negativ mit den gleitenden Durchschnitten, \
                was bedeutet: stabile Trends weisen geringere Volatilität auf, während chaotische Marktphasen \
                mit stark schwankenden Preisen einhergehen. → Die Features erklären Marktphasen gut, aber die \
                Tagesrichtung nur sehr begrenzt."
                )

        st.markdown("### 📊 Häufigkeit der Markt-Regime")
        st.plotly_chart(create_regime_frequency_bar_chart(df_master), use_container_width=True)

        st.caption(
            "Analyse: Der Markt ist klar von Aufwärtstrends geprägt. Mit **≈ 2.356 Bull-Tagen** dominiert das Bull-Regime "
            "den größten Teil der Historie. Das **Sideways-Regime tritt mit nur 380 Tagen** deutlich seltener auf, "
            "während das **Bear-Regime mit 1.155 Tagen** zwar kürzer ist, aber oft starke Abwärtsbewegungen enthält. "
            "Diese asymmetrische Verteilung zeigt: Bitcoin verbringt die meiste Zeit im Aufwärtstrend, während "
            "Bärenmärkte kompakt, aber wirtschaftlich besonders relevant sind. → Diese Regimeverteilung bildet "
            "die Grundlage für die spätere ML-Clusteranalyse."
)


        st.markdown("---")
        
        
        
        # -----------------------------------------------------------------
        # 5.2 THESENPRÜFUNG: BEWEISFÜHRUNG DURCH VISUALISIERUNG
        # -----------------------------------------------------------------
        st.subheader("5.2 Empirische Prüfung der Zentralen Thesen")
        st.markdown("Die folgenden Visualisierungen dienen der direkten Prüfung der wichtigsten Hypothesen zur Marktstruktur und Modellgüte. Der Erklärungstext ist direkt unter dem jeweiligen Plot platziert.")
        st.markdown("---")

        # -----------------------------------------------------------------
        # THESEN BLOCK 1: These 1 (Regime-Unterschiede) – Histogramm Version
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 1: Unterschiedliche Renditemuster in den Markt-Regimen")
        st.markdown("##### Histogramme: Verteilung der täglichen Renditen nach Regime")

        return_regime_hist = create_return_by_regime_histograms(df_master)
        st.plotly_chart(return_regime_hist, use_container_width=True)
        
        st.caption(
            "Analyse: Die Histogramme zeigen, dass Markt-Regime ökonomisch real sind: "
            "Bear = überwiegend negativ, Sideways = neutral um die Null-Linie, Bull = überwiegend positiv. "
            "→ These 1 bestätigt – die Regime unterscheiden sich signifikant in ihrer Renditeverteilung.")
        
        st.info(
            "**Beweisidee:** Diese drei Histogramme zeigen die Verteilung der täglichen Renditen getrennt nach den "
            "drei Markt-Regimen: Bear, Sideways und Bull.\n\n"
            "➡️ **Bear-Phasen:** Deutlich mehr negative Renditetage – Schwerpunkt links im negativen Bereich.\n"
            "➡️ **Sideways-Phasen:** Enge Verteilung um die Null-Linie – wenig Trend, geringe Dynamik.\n"
            "➡️ **Bull-Phasen:** Mehrheitlich positive Renditetage – linke Seite schmal, rechte Seite ausgeprägt.\n\n"
            "➡️ **Interpretation:** Die Histogramme zeigen klar erkennbare Muster: Bärenmärkte verlieren statistisch Geld, "
            "Seitwärtsmärkte verlaufen stabil ohne große Ausschläge, und Bullenmärkte generieren überwiegend positive Renditen.\n"
            "➡️ **Fazit:** These 1 wird bestätigt – die drei Markt-Regime unterscheiden sich signifikant in der Form und "
            "Lage ihrer Renditeverteilung."
        )

        st.markdown("---")

        # -----------------------------------------------------------------
        # THESEN BLOCK 2: These 2 (Cluster-Phasen)
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 2: ML-Cluster bilden unterschiedliche Marktphasen ab")
        st.markdown("##### Violinplot: Renditeverteilung nach Cluster-ID")

        st.caption(
            "Cluster-Interpretation: 🐂 **Bull-Markt = Cluster 0**, ➖ **Sideways = Cluster 1**, 🐻 **Bear-Markt = Cluster 2**")

        cluster_violin_fig = create_return_by_cluster_violinplot(df_master)
        st.plotly_chart(cluster_violin_fig, use_container_width=True)
        
        st.caption(
            "Analyse: Die Verteilungen zeigen klar unterschiedliche Marktphasen: "
            "Cluster 0 besitzt breite, positiv verschobene Renditen (Bull), "
            "Cluster 1 zeigt kompakte, neutrale Verteilungen (Sideways), "
            "und Cluster 2 besitzt linksschiefe, negative Renditen (Bear). "
            "→ These 2 bestätigt: ML-Cluster erkennen realwirtschaftliche Marktphasen.")


        st.info(
            "**Beweis:** Das ML-Clustering trennt den Markt in unterschiedliche Phasen, "
            "die klar unterschiedliche Renditemuster besitzen.\n\n"
            "➡️ Cluster mit hoher Volatilität zeigen breitere Verteilungen.\n"
            "➡️ Trendphasen zeigen verschobene Renditezentren.\n"
            "➡️ Ruhige Marktphasen haben schmale, kompakte Verteilungen.\n\n"
            "➡️ **Fazit:** These 2 wird bestätigt – die ML-Cluster spiegeln realwirtschaftlich unterschiedliche Marktphasen wider."
        )

        st.markdown("---")

        # -----------------------------------------------------------------
        # THESEN BLOCK 3: These 3 (Hitrate pro Cluster)
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 3: Modellleistung unterscheidet sich zwischen den ML-Clustern")
        st.markdown("##### Balkendiagramm: Trefferquote pro Cluster")

        hitrate_cluster_fig = create_hitrate_by_cluster_bar_chart(df_master)
        st.plotly_chart(hitrate_cluster_fig, use_container_width=True)
                # ⭐ Analyse in kleiner Schrift direkt zum Chart
        st.caption(
            "Analyse: Die Trefferquote des Direction-Modells unterscheidet sich deutlich zwischen den ML-Clustern. "
            "Cluster 1 liefert die höchste Trefferquote (~75%) und repräsentiert strukturierte Trendphasen, in denen Richtungsprognosen leicht fallen. "
            "Cluster 2 liegt im Mittelfeld (~65%). In Cluster 0 sinkt die Trefferquote auf nahezu Zufallsniveau (~52%) – typisch für unklare Seitwärts- oder Umschwungphasen. "
            "→ These 3 bestätigt: Die Modellqualität ist stark vom Marktumfeld abhängig.")

        st.info(
            "**Beweisidee:** Die Trefferquote des Direction-Modells variiert je nach Marktphase.\n\n"
            "➡️ **Trendstarke Cluster (z. B. Bull)** erreichen höhere Trefferquoten, "
            "da Aufwärts- oder Abwärtsbewegungen klarer erkennbar sind.\n"
            "➡️ **Seitwärts-Cluster** liegen näher bei ~50% – hier ist der Markt unstrukturiert, "
            "und die Richtung lässt sich schlechter prognostizieren.\n\n"
            "➡️ **Fazit:** These 3 wird bestätigt – die Modellleistung hängt signifikant davon ab, "
            "in welchem ML-Cluster (und damit Marktumfeld) wir uns befinden."
        )

        # ⭐ Zusatzinfo: Zuordnung der Cluster zu Regimen (gleiche Logik wie bei These 2)
        st.caption(
            "Cluster-Interpretation: 🐂 **Bull-Markt = Cluster 0**, ➖ **Sideways = Cluster 1**, 🐻 **Bear-Markt = Cluster 2**"
        )

        st.markdown("---")

        # # -----------------------------------------------------------------
        # # THESEN BLOCK 4: These 4 (Cluster vs. Regime Mapping)
        # # -----------------------------------------------------------------
        # st.markdown("#### 🎯 Prüfung These 4: ML-Cluster korrelieren mit den Markt-Regimen")
        # st.markdown("##### Heatmap: Wie gut mappt Cluster_3 auf Bear / Sideways / Bull?")

        # cluster_regime_fig = create_cluster_regime_heatmap(df_master)
        # st.plotly_chart(cluster_regime_fig, use_container_width=True)

        # st.info(
        #     "**Beweisidee:** Diese Heatmap zeigt, wie häufig jeder ML-Cluster in einem bestimmten Markt-Regime "
        #     "(Bear, Sideways, Bull) vorkommt. Die dunkelsten Felder markieren dabei die höchsten Häufigkeiten.\n\n"
        #     "➡️ **Starke Korrelation:** Wenn ein Cluster überwiegend in einem einzigen Regime vorkommt, erkennt das ML-Modell "
        #     "die gleiche Struktur wie die klassischen Marktregime – nur datengetriebener.\n"
        #     "➡️ **Feinere Granularität:** ML-Cluster unterscheiden teilweise Unterphasen innerhalb eines Regimes "
        #     "(z. B. verschiedene Bull-Intensitäten oder unterschiedliche Seitwärtsstrukturen).\n\n"
        #     "➡️ **Fazit:** These 4 wird bestätigt – die ML-Cluster sind nicht nur zufällige Gruppierungen, sondern bilden die "
        #     "Markt-Regime präzise ab und liefern gleichzeitig zusätzliche Detailtiefe."
        # )

        # # 🟦 Zusatzinfo: Zuordnung der Cluster zu Marktregimen (sofern ermittelt)
        # st.caption(
        #     "Cluster-Interpretation (vereinfachte Zuordnung): "
        #     "🐻 **Bear = Cluster 2**, ➖ **Sideways = Cluster 1**, 🐂 **Bull = Cluster 0**. "
        #     "→ ML-Cluster bilden die Regime weitgehend konsistent nach."
        # )

        # # 📊 Analyse-Fußnote – sehr kompakt & klein
        # st.caption(
        #     "Analyse: Die Heatmap zeigt eine klare diagonale Struktur – jeder ML-Cluster konzentriert sich hauptsächlich "
        #     "auf ein Regime. Das bestätigt die statistische Abbildung der Marktphasen durch das Clustering. "
        #     "→ ML-Cluster = datengetriebene Regime."
        # )

        # st.markdown("---")

        
        # -----------------------------------------------------------------
        # THESEN BLOCK 5: These 5 (Volatilität vs. neg. Rendite)
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 5: Hohe Marktvolatilität führt statistisch zu schlechteren Renditen")
        st.markdown("##### Scatter Plot: Rendite vs. Volatilität (30 Tage)")

        return_vol_fig = create_return_vs_volatility_scatter(df_master)
        st.plotly_chart(return_vol_fig, use_container_width=True)
        # Kleine kompakte Analyse (Caption)
        st.caption(
            "Analyse: Bei niedriger Volatilität bleiben Renditen stabil um 0 %. Steigende Volatilität führt dazu, "
            "dass die Punktwolke sichtbar nach unten ausfranst. Besonders im Bear-Regime dominieren dann negative Ausschläge."
        )

        st.info(
            "**Beweisidee:** Dieser Scatter Plot zeigt, wie sich tägliche Renditen (`Return`) unter unterschiedlichen "
            "Volatilitätsbedingungen (`Volatility30`) verhalten. Die graue Trendlinie (OLS) verdeutlicht den "
            "**durchschnittlichen statistischen Zusammenhang** zwischen beiden Größen.\n\n"

            "➡️ **Leicht fallende Trendlinie:** Die Linie neigt sich sichtbar nach unten. Das bedeutet: "
            "**Mit steigender Volatilität sinken die durchschnittlichen Renditen.** Hohe Volatilität ist also ein "
            "Frühindikator für Stress im Markt.\n\n"

            "➡️ **Grafikanalyse:** Die Punktwolke ist breit gestreut, aber man erkennt klare Muster:\n"
            "• **Ab ca. 0.05 Volatilität** verschieben sich viele Punkte deutlich in den negativen Renditebereich.\n"
            "• **Bärenmärkte (rot)** sammeln sich rechts unten – also bei hoher Volatilität und gleichzeitig klar negativen Renditen.\n"
            "• **Bullenmärkte (grün)** liegen häufiger über der Null-Linie, aber überwiegend bei mittlerer Volatilität – "
            "Bull-Märkte funktionieren typischerweise besser in ruhigeren Phasen.\n"
            "• **Seitwärtsmärkte (gelb)** konzentrieren sich eng um 0 % Rendite – typisch für neutrale, "
            "weniger trendstarke Marktphasen.\n\n"

            "➡️ **Fazit:** These 5 wird bestätigt – **je chaotischer der Markt, desto häufiger und tiefer sind Verluste**. "
            "Volatilität wirkt als Risiko- und Stressbarometer des Marktes."
        )


        st.markdown("---")

        
        # -----------------------------------------------------------------
        # THESEN BLOCK 6: These 6 (MA-Differenz)
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 6: Gleitende Durchschnitte erklären Trendphasen klar und intuitiv")
        st.markdown("##### Boxplot: Unterschied zwischen kurzem und langem Durchschnitt (MA50 - MA200) nach Regime")

        ma_diff_fig = create_ma_diff_by_regime_boxplot(df_master) 
        st.plotly_chart(ma_diff_fig, use_container_width=True)

        # Kurze kompakte Analyse
        st.caption(
            "Analyse: In Bärenmärkten liegt der kurzfristige Durchschnitt stark unter dem langfristigen – "
            "in Bullenmärkten deutlich darüber. Seitwärtsphasen liegen dazwischen und schwanken breiter."
        )

        st.info(
            "**Beweisidee (einfach erklärt):** Ein *gleitender Durchschnitt* ist nichts anderes als ein durchschnittlicher Preis "
            "über einen bestimmten Zeitraum. Der **kurze Durchschnitt (50 Tage)** reagiert schneller, der **lange Durchschnitt (200 Tage)** "
            "reagiert langsamer.\n\n"
            
            "➡️ **Was bedeutet das?** Wenn der kurzfristige Durchschnitt über dem langfristigen liegt, bewegt sich der Markt tendenziell "
            "aufwärts (Bull-Trend). Liegt er darunter, dominiert ein Abwärtstrend (Bear). Genau diese Logik nutzt man seit Jahrzehnten, "
            "um Trends zu erkennen.\n\n"
            
            "### 🔍 Was zeigt die Grafik?\n"
            "• **Bear-Regime (rot):** Die Werte liegen klar **unter Null** – der kurzfristige Preisverlauf liegt deutlich "
            "unter dem langfristigen. Das ist typisch für längere Abwärtsphasen.\n"
            "• **Sideways-Regime (gelb):** Die Werte liegen **über Null**, aber schwanken stark. Der Markt hat keine klare "
            "Richtung – mal über, mal unter dem langfristigen Trend.\n"
            "• **Bull-Regime (grün):** Sehr enge Werte **über Null** – der kurzfristige Trend liegt stabil über dem langfristigen. "
            "Der Markt steigt ruhig und relativ konstant.\n\n"
            
            "➡️ **Visuelle Interpretation:** Die graue Null-Linie ist die „Trendgrenze“. "
            "**Bärenmärkte liegen darunter**, **Bullenmärkte darüber**, **Seitwärtsmärkte drumherum**.\n\n"

            "➡️ **Fazit:** These 6 wird klar bestätigt – der Unterschied zwischen kurzem und langem Durchschnitt ist ein "
            "intuitives und extrem zuverlässiges Signal, um Marktphasen sauber zu unterscheiden."
        )

        st.markdown("---")


        # -----------------------------------------------------------------
        # THESEN BLOCK 7: These 7 (Direction Prediction ist extrem schwer)
        # -----------------------------------------------------------------
        st.markdown("#### 🎯 Prüfung These 7: ML kann Trends erkennen – aber nicht zuverlässig die Richtung des nächsten Tages")
        st.markdown("##### Balkendiagramm: Modellgenauigkeit (F1-Score) für Up/Down-Vorhersage")

        direction_fig = create_direction_model_performance_plot()
        st.plotly_chart(direction_fig, use_container_width=True)
        
        st.caption(
            "**Analyse:** Alle Modelle erreichen nur etwa **45–50 % F1-Score** und liegen damit "
            "**kaum über dem Zufallsniveau**. Das zeigt: Selbst moderne ML-Algorithmen können die "
            "**Tagesrichtung (Up/Down)** des Bitcoin-Kurses **nicht zuverlässig** vorhersagen. "
            "Kurzfristige Marktbewegungen sind überwiegend **zufällig**, stark **newsgetrieben** "
            "und schwer modellierbar. → **Bestätigung von These 7:** ML erkennt Marktstrukturen "
            "(Regime) sehr gut – aber bei der Frage *„Steigt der Kurs morgen?“* stößt es klar "
            "an seine Grenzen."
)


        st.info(
            "**Beweisidee:** Hier sehen wir die tatsächliche Leistung verschiedener ML-Modelle bei der Vorhersage, "
            "ob der Bitcoin-Kurs am nächsten Tag **steigt (Up)** oder **fällt (Down)**.\n\n"

            "➡️ **F1-Scores liegen nur zwischen ~50–60%.** Das bedeutet: Die Modelle sind nur wenig besser als Zufall.\n\n"

            "➡️ **Warum ist das so?** Tagesbewegungen sind extrem zufällig und werden von Nachrichten, Liquidität, "
            "großen Tradern und globalen Ereignissen dominiert. Diese Faktoren sieht das Modell nicht.\n\n"

            "➡️ **Wichtiger Lerneffekt:** ML kann **Marktphasen** hervorragend erkennen (siehe Regime), "
            "aber **Tagesrichtung** ist statistisch kaum vorhersagbar.\n\n"

            "➡️ **Fazit:** These 7 zeigt bewusst die Grenzen von Machine Learning: "
            "**Strukturen ja — Hellsehen nein.**"
        )

        st.markdown("---")
        
                # -----------------------------------------------------------------
        # # 5.3 ROHDATEN-VORSCHAU
        # # -----------------------------------------------------------------
        # st.subheader("Rohdaten-Vorschau (Features)")
        # st.markdown("Anzeige der letzten Handelstage zur direkten explorativen Überprüfung von Features und dem zugewiesenen Markt-Regime.")
        
        # st.dataframe(df_master.tail(10).style.apply(highlight_focus_day, axis=1, focus_date_str=fokus_tag), use_container_width=True)

