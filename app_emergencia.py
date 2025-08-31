# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# ========= Page config (sin enlaces de menú) =========
st.set_page_config(
    page_title="PREDICCION EMERGENCIA AGRICOLA LOLIUM SP",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

# ========= CSS para ocultar UI no deseada =========
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__1QSob {visibility: hidden;}
    .st-emotion-cache-9aoz2h {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# =================== Utilidades ===================
@st.cache_data(show_spinner=False)
def obtener_colores(niveles: pd.Series) -> np.ndarray:
    # vectorizado y sin dependencias externas
    mapa = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
    return niveles.map(mapa).fillna("gray").to_numpy()

def validar_columnas(df: pd.DataFrame) -> tuple[bool, str]:
    req = {"Julian_days", "TMAX", "TMIN", "Prec", "Fecha"}
    faltan = req - set(df.columns)
    if faltan:
        return False, f"Faltan columnas: {', '.join(sorted(faltan))}"
    return True, ""

def detectar_fuera_rango(X_real: np.ndarray, input_min: np.ndarray, input_max: np.ndarray) -> bool:
    out = (X_real < input_min) | (X_real > input_max)
    return bool(np.any(out))

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out, low=0.02, medium=0.079):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        # Orden esperado: [Julian_days, TMAX, TMIN, Prec]
        self.input_min = np.array([1, 0, -7, 0], dtype=float)
        self.input_max = np.array([300, 41, 25.5, 84], dtype=float)
        self.low_thr = float(low)
        self.med_thr = float(medium)

    @staticmethod
    def tansig(x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        # 2*(x-min)/(max-min)-1  -> vectorizado
        return 2.0 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1.0

    @staticmethod
    def desnormalizar_salida(y_norm, ymin=-1.0, ymax=1.0):
        # map [-1,1] -> [0,1]
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_matrix(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Versión vectorizada para todo el lote (evita loop Python).
        X_norm: (n, 4)
        """
        # capa oculta: a1 = tanh(x*IW + bias_IW)
        # Nota: IW en tu guardado es consistente con z1 = IW.T @ x_norm + bias_IW (por muestra).
        # Para vectorizar, replicamos ese cálculo: z1_i = x_i @ IW + bias_IW
        z1 = X_norm @ self.IW + self.bias_IW  # (n, hidden)
        a1 = self.tansig(z1)
        # salida: z2 = a1 @ LW.T + bias_out
        z2 = a1 @ self.LW.T + self.bias_out  # (n, 1)
        return self.tansig(z2).ravel()       # (n,)

    def _clasificar_vec(self, valores: np.ndarray) -> np.ndarray:
        # Clasificación vectorizada
        out = np.empty_like(valores, dtype=object)
        out[valores < self.low_thr] = "Bajo"
        mask_medio = (valores >= self.low_thr) & (valores <= self.med_thr)
        out[mask_medio] = "Medio"
        out[valores > self.med_thr] = "Alto"
        return out

    def predict_df(self, X_real: np.ndarray) -> pd.DataFrame:
        X_norm = self.normalize_input(X_real)
        emerrel_pred = self._predict_matrix(X_norm)                       # [-1,1]
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)         # [0,1]
        emer_acum = np.cumsum(emerrel_desnorm) / 8.05                     # EMEAC normalizado
        emerrel_diff = np.diff(emer_acum, prepend=0.0)
        riesgo = self._clasificar_vec(emerrel_diff)
        return pd.DataFrame(
            {"EMERREL(0-1)": emerrel_diff, "Nivel_Emergencia_relativa": riesgo}
        )

# =================== Carga de pesos y datos ===================
@st.cache_resource(show_spinner=False)
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

@st.cache_resource(show_spinner=False)
def build_model(base_dir: Path) -> PracticalANNModel:
    IW, bias_IW, LW, bias_out = load_weights(base_dir)
    return PracticalANNModel(IW, bias_IW, LW, bias_out)

CSV_URL_PAGES = "https://PREDWEEM.github.io/ANN/meteo_daily.csv"
CSV_URL_RAW   = "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_daily.csv"

@st.cache_data(ttl=900, show_spinner=False)
def load_public_csv() -> tuple[pd.DataFrame, str]:
    last_err = None
    usecols = ["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]
    dtypes = {"Julian_days": "int16", "TMAX": "float32", "TMIN": "float32", "Prec": "float32"}
    for url in (CSV_URL_PAGES, CSV_URL_RAW):
        try:
            df = pd.read_csv(
                url,
                usecols=usecols,
                dtype=dtypes,
                parse_dates=["Fecha"],
                dayfirst=False,
            )
            ok, msg = validar_columnas(df)
            if not ok:
                raise ValueError(msg)
            # Orden único por Fecha (ya es suficiente para más adelante)
            df = df.sort_values("Fecha", kind="stable").reset_index(drop=True)
            return df, url
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo leer el CSV público. Último error: {last_err}")

# =================== UI ===================
st.title("PREDICCION EMERGENCIA AGRICOLA LOLIUM SP")

# Sidebar con formulario para evitar re-ejecución constante
with st.sidebar:
    st.header("Configuración")
    with st.form("cfg"):
        umbral_usuario = st.number_input(
            "Umbral de EMEAC para 100%",
            min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f"
        )
        mostrar_fuera_rango = st.checkbox(
            "Avisar datos fuera de rango de entrenamiento", value=False
        )
        aplicar = st.form_submit_button("Aplicar")

# Preserva estado para no recomputar si no cambia
if aplicar or "umbral_usuario" not in st.session_state:
    st.session_state["umbral_usuario"] = float(umbral_usuario)
    st.session_state["mostrar_fuera_rango"] = bool(mostrar_fuera_rango)

umbral_usuario = st.session_state["umbral_usuario"]
mostrar_fuera_rango = st.session_state["mostrar_fuera_rango"]

# Cargar pesos del modelo
try:
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    modelo = build_model(base)
except FileNotFoundError:
    st.error(
        "Error al cargar archivos del modelo. Verifique que IW.npy, bias_IW.npy, LW.npy y bias_out.npy estén junto al script."
    )
    st.stop()

# =================== Datos (CSV público) ===================
try:
    df, _src = load_public_csv()
except Exception:
    st.error("No se pudo leer el CSV público. Intente más tarde o revise la fuente.")
    st.stop()

# =================== Predicción (cacheada por datos) ===================
@st.cache_data(show_spinner=False)
def run_prediction_block(df_in: pd.DataFrame, modelo: PracticalANNModel) -> pd.DataFrame:
    # Evita copiar todo: sólo las columnas necesarias en NumPy directamente
    X_real = df_in[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float, copy=False)
    pred = modelo.predict_df(X_real)
    pred["Fecha"] = df_in["Fecha"].to_numpy(copy=False)
    pred["Julian_days"] = df_in["Julian_days"].to_numpy(copy=False)
    # columnas acumuladas generales (pueden reutilizarse si hiciera falta)
    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
    # MA5 general (luego recalculamos en rango)
    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    return pred

pred = run_prediction_block(df, modelo)

# =================== Ventana 1/feb → 1/sep (reinicio) ===================
years = pred["Fecha"].dt.year.unique()
yr = int(years[0]) if len(years) == 1 else int(np.min(years))
fecha_inicio_rango = pd.Timestamp(year=yr, month=2, day=1)
fecha_fin_rango    = pd.Timestamp(year=yr, month=9, day=1)

mask = (pred["Fecha"] >= fecha_inicio_rango) & (pred["Fecha"] <= fecha_fin_rango)
pred_vis = pred.loc[mask].copy()

if pred_vis.empty:
    st.warning(f"No hay datos entre {fecha_inicio_rango.date()} y {fecha_fin_rango.date()} para la fuente.")
    st.stop()

# Recalcular acumulados y % EMEAC dentro del rango (reiniciados)
emerrel = pred_vis["EMERREL(0-1)"].to_numpy()
emerrel_acum_rango = np.cumsum(emerrel)
pred_vis["EMERREL acumulado (reiniciado)"] = emerrel_acum_rango

inv_min = 1.0 / 1.2
inv_max = 1.0 / 3.0
inv_usr = 1.0 / umbral_usuario

pred_vis["EMEAC (0-1) - mínimo (rango)"]    = emerrel_acum_rango * inv_min
pred_vis["EMEAC (0-1) - máximo (rango)"]    = emerrel_acum_rango * inv_max
pred_vis["EMEAC (0-1) - ajustable (rango)"] = emerrel_acum_rango * inv_usr

pred_vis["EMEAC (%) - mínimo (rango)"]    = pred_vis["EMEAC (0-1) - mínimo (rango)"] * 100.0
pred_vis["EMEAC (%) - máximo (rango)"]    = pred_vis["EMEAC (0-1) - máximo (rango)"] * 100.0
pred_vis["EMEAC (%) - ajustable (rango)"] = pred_vis["EMEAC (0-1) - ajustable (rango)"] * 100.0

# Media móvil dentro del rango (cálculo único)
pred_vis["EMERREL_MA5_rango"] = (
    pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
)

# Validación de rangos (opcional)
if mostrar_fuera_rango:
    X_real_check = df.loc[mask, ["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float, copy=False)
    if detectar_fuera_rango(X_real_check, modelo.input_min, modelo.input_max):
        st.info(
            f"⚠️ Hay valores fuera del rango de entrenamiento "
            f"({modelo.input_min.tolist()} a {modelo.input_max.tolist()})."
        )

# =================== GRÁFICOS ===================
# --------- Gráfico 1: EMERGENCIA RELATIVA DIARIA ---------
st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")

colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

fig_er = go.Figure()
fig_er.add_bar(
    x=pred_vis["Fecha"],
    y=pred_vis["EMERREL(0-1)"],
    marker=dict(color=colores_vis),
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}",
    customdata=pred_vis["Nivel_Emergencia_relativa"],
    name="EMERREL (0-1)",
)

fig_er.add_trace(go.Scatter(
    x=pred_vis["Fecha"],
    y=pred_vis["EMERREL_MA5_rango"],
    mode="lines",
    name="Media móvil 5 días (rango)",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>",
    line=dict(width=2)
))

# Líneas de referencia (mantener livianas)
low_thr = float(modelo.low_thr)
med_thr = float(modelo.med_thr)

fig_er.add_trace(go.Scatter(
    x=[fecha_inicio_rango, fecha_fin_rango],
    y=[low_thr, low_thr],
    mode="lines",
    line=dict(dash="dot"),
    name=f"Bajo (≤ {low_thr:.3f})",
    hoverinfo="skip"
))
fig_er.add_trace(go.Scatter(
    x=[fecha_inicio_rango, fecha_fin_rango],
    y=[med_thr, med_thr],
    mode="lines",
    line=dict(dash="dot"),
    name=f"Medio (≤ {med_thr:.3f})",
    hoverinfo="skip"
))
# leyenda para "Alto"
fig_er.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="lines",
    line=dict(dash="dot"),
    name=f"Alto (> {med_thr:.3f})",
    hoverinfo="skip",
    showlegend=True
))

fig_er.update_layout(
    xaxis_title="Fecha",
    yaxis_title="EMERREL (0-1)",
    hovermode="x unified",
    legend_title="Referencias",
    height=600,
    margin=dict(l=10, r=10, t=30, b=10)
)
fig_er.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
fig_er.update_yaxes(rangemode="tozero")

st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

# --------- Gráfico 2: EMEAC (rango) ---------
st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")

fig = go.Figure()

# banda entre mínimo y máximo (dos trazas, sin ancho de línea)
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"],
    y=pred_vis["EMEAC (%) - máximo (rango)"],
    mode="lines",
    line=dict(width=0),
    name="Máximo (reiniciado)",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"],
    y=pred_vis["EMEAC (%) - mínimo (rango)"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    name="Mínimo (reiniciado)",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
))

# umbral ajustable con línea más visible
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"],
    y=pred_vis["EMEAC (%) - ajustable (rango)"],
    mode="lines",
    name="Umbral ajustable (reiniciado)",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>",
    line=dict(width=2.5)
))

# líneas guía
for nivel in (25, 50, 75, 90):
    fig.add_hline(y=nivel, line_dash="dash", opacity=0.5, annotation_text=f"{nivel}%")

fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="EMEAC (%)",
    yaxis=dict(range=[0, 100]),
    hovermode="x unified",
    legend_title="Referencias",
    height=560,
    margin=dict(l=10, r=10, t=30, b=10)
)
fig.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# =================== Tabla y descarga ===================
st.subheader(f"Resultados (1/feb → 1/sep) - MeteoBahia_CSV")

col_emeac = "EMEAC (%) - ajustable (rango)"
tabla_base = pred_vis.loc[:, ["Fecha", "Julian_days", "Nivel_Emergencia_relativa", col_emeac]].rename(
    columns={"Julian_days": "Día juliano", "Nivel_Emergencia_relativa": "Nivel de EMERREL", col_emeac: "EMEAC (%)"}
)

nivel_emoji = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
# map vectorizado
tabla_display = tabla_base.copy()
tabla_display["Nivel de EMERREL"] = tabla_display["Nivel de EMERREL"].map(lambda x: f"{nivel_emoji.get(x, '')} {x}")

st.dataframe(tabla_display, use_container_width=True)

csv = tabla_base.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar resultados (rango) - MeteoBahia_CSV",
    csv,
    "MeteoBahia_CSV_resultados_rango.csv",
    "text/csv"
)
