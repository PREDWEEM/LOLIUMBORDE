# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# ========== Configuración de la página ==========
st.set_page_config(
    page_title="PREDICCIÓN EMERGENCIA AGRÍCOLA LOLIUM SP",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"],
.viewerBadge_container__1QSob, .st-emotion-cache-9aoz2h, .stAppDeployButton {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ========== Clase del modelo ==========
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
        emer_ac = emerrel_cumsum / 8.05
        emerrel_diff = np.diff(emer_ac, prepend=0)
        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

# ========== Cargar pesos ==========
@st.cache_data
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

# ========== Cargar CSV público ==========
CSV_URL = "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_daily.csv"

@st.cache_data(ttl=900)
def load_public_csv():
    df = pd.read_csv(CSV_URL, parse_dates=["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df

# ========== Cargar XML de pronóstico (8 días) ==========
@st.cache_data(ttl=1800)
def load_forecast_from_xml(limit_days=8):
    url = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
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

# ========== Interfaz ==========
st.title("🌾 PREDICCIÓN DE EMERGENCIA AGRÍCOLA - LOLIUM SP")

with st.sidebar:
    st.header("Configuración")
    umbral_usuario = st.number_input("Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0, value=2.70, step=0.01)

# ========== Cargar pesos ==========
try:
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    IW, bias_IW, LW, bias_out = load_weights(base)
except Exception:
    st.error("Error al cargar archivos del modelo (IW.npy, etc.)")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# ========== Cargar y unir datos ==========
try:
    df_hist = load_public_csv()
    df_fore = load_forecast_from_xml(limit_days=8)
    df_total = pd.concat([df_hist, df_fore], ignore_index=True).drop_duplicates("Fecha").sort_values("Fecha").reset_index(drop=True)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ========== Procesar ==========
X_real = df_total[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
pred = modelo.predict(X_real)

df_pred = df_total.copy()
df_pred["EMERREL(0-1)"] = pred["EMERREL(0-1)"]
df_pred["Nivel_Emergencia_relativa"] = pred["Nivel_Emergencia_relativa"]
df_pred["EMERREL acumulado"] = df_pred["EMERREL(0-1)"].cumsum()
df_pred["EMEAC (%)"] = (df_pred["EMERREL acumulado"] / umbral_usuario) * 100

# ========== Filtrar rango Feb-Oct ==========
yr = df_pred["Fecha"].dt.year.min()
start, end = pd.Timestamp(yr, 2, 1), pd.Timestamp(yr, 10, 1)
df_vis = df_pred[(df_pred["Fecha"] >= start) & (df_pred["Fecha"] <= end)].copy()
df_vis["MA5"] = df_vis["EMERREL(0-1)"].rolling(5, min_periods=1).mean()

# ========== Gráfico 1 ==========
st.subheader("📊 EMERGENCIA RELATIVA DIARIA (1/feb → 1/oct)")
color_map = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
colores = df_vis["Nivel_Emergencia_relativa"].map(color_map).fillna("gray")

fig = go.Figure()
fig.add_bar(x=df_vis["Fecha"], y=df_vis["EMERREL(0-1)"], marker=dict(color=colores))
fig.add_trace(go.Scatter(x=df_vis["Fecha"], y=df_vis["MA5"], mode="lines", name="Media móvil 5 días"))
fig.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", height=500)
st.plotly_chart(fig, use_container_width=True)

# ========== Gráfico 2 ==========
st.subheader("📈 EMEAC (%) ACUMULADO")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_vis["Fecha"], y=df_vis["EMEAC (%)"], mode="lines", name="EMEAC (%)"))
for y in [25, 50, 75, 90]:
    fig2.add_hline(y=y, line_dash="dash", annotation_text=f"{y}%", opacity=0.4)
fig2.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", height=450, yaxis=dict(range=[0, 100]))
st.plotly_chart(fig2, use_container_width=True)

# ========== Tabla ==========
st.subheader("📋 Tabla de resultados")
tabla = df_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa", "EMEAC (%)"]].copy()
emoji = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
tabla["Nivel_Emergencia_relativa"] = tabla["Nivel_Emergencia_relativa"].map(lambda x: f"{emoji.get(x, '')} {x}")
st.dataframe(tabla, use_container_width=True)

# ========== Descargar ==========
csv = tabla.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV", csv, "resultados_emergencia.csv", "text/csv")
