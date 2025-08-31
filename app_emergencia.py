# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# ========= Page config =========
st.set_page_config(
    page_title="PREDICCION EMERGENCIA AGRICOLA LOLIUM SP",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

st.markdown("""
    <style>
    #MainMenu, footer, header [data-testid="stToolbar"],
    .viewerBadge_container__1QSob, .st-emotion-cache-9aoz2h,
    .stAppDeployButton {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out, low=0.02, medium=0.079):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
        self.low_thr = low
        self.med_thr = medium

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalizar_salida(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def _clasificar(self, valor):
        if valor < self.low_thr:
            return "Bajo"
        elif valor <= self.med_thr:
            return "Medio"
        else:
            return "Alto"

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)
        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

# =================== Carga de pesos ANN ===================
@st.cache_data(show_spinner=False)
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

# =================== Carga de datos históricos CSV ===================
CSV_URL_PAGES = "https://PREDWEEM.github.io/ANN/meteo_daily.csv"
CSV_URL_RAW   = "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_daily.csv"

@st.cache_data(ttl=900)
def load_public_csv():
    for url in (CSV_URL_PAGES, CSV_URL_RAW):
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            required = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
            if not required.issubset(df.columns):
                raise ValueError(f"Faltan columnas: {required - set(df.columns)}")
            return df.sort_values("Fecha").reset_index(drop=True), url
        except Exception:
            continue
    raise RuntimeError("No se pudo leer el CSV público.")

# =================== Carga del pronóstico XML (8 días) ===================
@st.cache_data(ttl=1800)
def load_forecast_from_xml(limit_days=8):
    url = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        records = []
        for i, item in enumerate(root.findall(".//item")):
            if i >= limit_days:
                break
            fecha = datetime.strptime(item.findtext("fecha"), "%Y-%m-%d")
            tmin = float(item.findtext("tmin", default="0"))
            tmax = float(item.findtext("tmax", default="0"))
            prec = float(item.findtext("precipitacion", default="0"))
            records.append({
                "Fecha": fecha,
                "Julian_days": fecha.timetuple().tm_yday,
                "TMAX": tmax,
                "TMIN": tmin,
                "Prec": prec
            })
        return pd.DataFrame(records).sort_values("Fecha").reset_index(drop=True)
    except Exception as e:
        st.warning(f"No se pudo cargar el pronóstico: {e}")
        return pd.DataFrame()

# =================== Configuración UI ===================
st.title("PREDICCIÓN DE EMERGENCIA AGRÍCOLA - LOLIUM SP")

with st.sidebar:
    st.header("Configuración")
    umbral_usuario = st.number_input("Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0, value=2.70, step=0.01)
    st.header("Validaciones")
    mostrar_fuera_rango = st.checkbox("Avisar datos fuera del rango de entrenamiento", value=False)

# =================== Cargar modelo ANN ===================
try:
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    IW, bias_IW, LW, bias_out = load_weights(base)
except FileNotFoundError:
    st.error("Faltan archivos del modelo (IW.npy, bias_IW.npy, etc).")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# =================== Cargar y unir datos ===================
dfs = []
try:
    df_auto, _ = load_public_csv()
    df_forecast = load_forecast_from_xml(limit_days=8)

    if not df_forecast.empty:
        df_total = pd.concat([df_auto, df_forecast], ignore_index=True)
        df_total = df_total.sort_values("Fecha").drop_duplicates("Fecha").reset_index(drop=True)
        dfs.append(("Histórico + Pronóstico (8 días)", df_total))
    else:
        dfs.append(("MeteoBahia_CSV", df_auto))
except Exception as e:
    st.error("No se pudo leer el CSV o el XML.")
    st.stop()

# =================== Procesamiento y visualización ===================
def obtener_colores(niveles):
    return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"}).fillna("gray")

def detectar_fuera_rango(X_real, input_min, input_max):
    out = (X_real < input_min) | (X_real > input_max)
    return bool(np.any(out))

for nombre, df in dfs:
    df = df.sort_values("Julian_days").reset_index(drop=True)
    X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    fechas = pd.to_datetime(df["Fecha"])

    if mostrar_fuera_rango and detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
        st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento.")

    pred = modelo.predict(X_real)
    pred["Fecha"] = fechas
    pred["Julian_days"] = df["Julian_days"]
    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
    pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

    yr = pred["Fecha"].dt.year.min()
    fecha_inicio = pd.Timestamp(year=yr, month=2, day=1)
    fecha_fin    = pd.Timestamp(year=yr, month=10, day=1)
    mask = (pred["Fecha"] >= fecha_inicio) & (pred["Fecha"] <= fecha_fin)
    pred_vis = pred.loc[mask].copy()

    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (%) - ajustable (rango)"] = (pred_vis["EMERREL acumulado (reiniciado)"] / umbral_usuario) * 100
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

    # ========== GRÁFICO 1: EMERREL DIARIO ==========
    st.subheader("EMERGENCIA RELATIVA DIARIA (1/feb → 1/oct) - BORDENAVE")
    fig_er = go.Figure()
    fig_er.add_bar(x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"], marker=dict(color=colores_vis.tolist()), name="EMERREL (0-1)", customdata=pred_vis["Nivel_Emergencia_relativa"], hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}")
    fig_er.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"], mode="lines", name="Media móvil 5 días", line=dict(width=2.5)))

    # Líneas de referencia
    low_thr = float(modelo.low_thr)
    med_thr = float(modelo.med_thr)
    fig_er.add_hline(y=low_thr, line_dash="dot", line_color="green", annotation_text=f"Bajo (≤ {low_thr:.3f})")
    fig_er.add_hline(y=med_thr, line_dash="dot", line_color="orange", annotation_text=f"Medio (≤ {med_thr:.3f})")
    fig_er.add_hline(y=0.15, line_dash="dot", line_color="red", annotation_text=f"Alto (> {med_thr:.3f})", opacity=0)

    fig_er.update_layout(
        xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
        hovermode="x unified", height=600, legend_title="Referencia"
    )
    st.plotly_chart(fig_er, use_container_width=True)

    # ========== GRÁFICO 2: EMEAC (%) acumulado ==========
    st.subheader("EMERGENCIA ACUMULADA - BORDENAVE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable (rango)"], mode="lines", name="EMEAC (%) - ajustable", line=dict(width=2.5)))
    for nivel in [25, 50, 75, 90]:
        fig.add_hline(y=nivel, line_dash="dash", opacity=0.5, annotation_text=f"{nivel}%")

    fig.update_layout(
        xaxis_title="Fecha", yaxis_title="EMEAC (%)",
        yaxis=dict(range=[0, 100]), hovermode="x unified", height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========== TABLA ==========
    st.subheader(f"Resultados (1/feb → 1/oct) - {nombre}")
    tabla = pred_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa", "EMEAC (%) - ajustable (rango)"]]
    tabla = tabla.rename(columns={"Julian_days": "Día juliano", "Nivel_Emergencia_relativa": "Nivel de EMERREL", "EMEAC (%) - ajustable (rango)": "EMEAC (%)"})
    nivel_emoji = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
    tabla["Nivel de EMERREL"] = tabla["Nivel de EMERREL"].map(lambda x: f"{nivel_emoji.get(x, '')} {x}")
    st.dataframe(tabla, use_container_width=True)

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(f"Descargar resultados - {nombre}", csv, f"{nombre}_resultados.csv", "text/csv")
