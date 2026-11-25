from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SENATI · PAVEMENT AI | Huánuco",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "pavimento_huanuco_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "pavement_lr_model.joblib"
PATOLOGIAS_PATH = BASE_DIR / "data" / "patologias_registradas.csv"
UPLOADS_DIR = BASE_DIR / "app" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PATOLOGIAS_SVG = [
    ("Grietas longitudinales", "M55 5 L48 35 58 60 50 95", "Fisura paralela al eje de la vía o junta."),
    ("Grietas transversales", "M5 50 L35 45 65 55 95 48", "Fisura perpendicular al eje de la losa."),
    ("Grietas de esquina", "M5 40 L40 5", "Fractura que intercepta juntas a <1.8 m."),
    ("Losa dividida", "M5 50 L50 40 95 60 M50 5 L50 95", "Intersección de fisuras en 4 o más partes."),
    ("Despostillamiento", "M5 48 L20 40 30 56 45 42 60 57 80 40 95 50", "Desprendimiento o astillado cerca de juntas."),
    ("Escalonamiento", "M5 45 H50 V60 H95", "Desnivel vertical entre losas adyacentes."),
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: linear-gradient(155deg, #f0f7fa 0%, #f5f3fb 50%, #fdf4f7 100%);
        color: #16253b;
    }

    /* Banner principal con degradado índigo-turquesa-coral luminoso */
    .hero-banner {
        background: linear-gradient(125deg, #1b3b6f 0%, #16697a 58%, #2a8c82 100%);
        color: #ffffff;
        padding: 34px 32px;
        border-radius: 22px;
        box-shadow: 0 18px 38px rgba(27, 59, 111, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 14px;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        line-height: 1.06;
    }
    .hero-subtitle {
        color: #e8f8f7;
        font-size: 1.02rem;
        margin-bottom: 14px;
    }
    .highway-strip {
        height: 36px;
        background: linear-gradient(180deg, #374b61 0%, #263547 100%);
        border-radius: 10px;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin: 12px 0;
    }
    .highway-strip::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 16px;
        right: 16px;
        transform: translateY(-50%);
        border-top: 3px dashed #ffd166;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .meta-pill {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.26);
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 9px;
    }

    /* Botones de navegación con alto contraste y toque coral en el activo */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.52rem 0.75rem !important;
        border: 1.5px solid #cbd9e8 !important;
        background-color: #ffffff !important;
        color: #16253b !important;
        transition: all 0.16s ease !important;
    }
    div.stButton > button:hover {
        border-color: #16697a !important;
        background-color: #ebf7f8 !important;
        color: #1b3b6f !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e76f51 0%, #f4a261 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 7px 18px rgba(231, 111, 81, 0.28) !important;
    }

    /* Campos de formulario claros y fáciles de leer */
    .stTextInput input, .stNumberInput input, .stDateInput input {
        background-color: #f8fbfe !important;
        color: #16253b !important;
        border: 1.5px solid #c5d6e8 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #f8fbfe !important;
        color: #16253b !important;
        border: 1.5px solid #c5d6e8 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span {
        color: #16253b !important;
    }

    /* Tarjetas blancas con borde superior turquesa-coral */
    .surface-card {
        background: #ffffff;
        border: 1px solid #dde7f2;
        border-top: 4px solid #16697a;
        border-radius: 20px;
        padding: 24px 26px;
        box-shadow: 0 10px 26px rgba(22, 37, 59, 0.05);
        margin-top: 12px;
        margin-bottom: 18px;
    }
    .section-tag {
        color: #e76f51;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-card {
        background: linear-gradient(180deg, #eff8fa 0%, #e1f1f5 100%);
        border: 1px solid #c2dfe7;
        border-radius: 15px;
        padding: 16px 18px;
        height: 100%;
    }
    .kpi-card .k-title {
        color: #16697a;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .kpi-card .k-desc {
        color: #16253b;
        font-size: 0.95rem;
        line-height: 1.45;
        margin: 0;
    }

    .pat-box {
        background: #f5f9fc;
        border: 1px solid #d5e4f0;
        border-radius: 15px;
        padding: 16px 12px;
        text-align: center;
        margin-bottom: 14px;
        transition: transform 0.16s ease, border-color 0.16s ease;
    }
    .pat-box:hover {
        transform: translateY(-2px);
        border-color: #e76f51;
    }
    .pat-box .pat-name {
        color: #16253b;
        font-weight: 700;
        font-size: 0.92rem;
        margin-top: 8px;
        display: block;
    }
    .pat-box .pat-sub {
        color: #51667d;
        font-size: 0.77rem;
        line-height: 1.35;
        margin-top: 4px;
        display: block;
    }

    .callout-amber {
        border-left: 4px solid #e76f51;
        background: #fff8f3;
        border-top: 1px solid #f7dfd4;
        border-right: 1px solid #f7dfd4;
        border-bottom: 1px solid #f7dfd4;
        color: #3d2217;
        padding: 13px 16px;
        border-radius: 11px;
        font-size: 0.92rem;
        margin-top: 12px;
    }

    /* Simulador en tono índigo-turquesa vibrante */
    .sim-panel {
        background: linear-gradient(145deg, #1b3b6f 0%, #16697a 100%);
        color: #ffffff;
        border-radius: 20px;
        padding: 24px;
        margin-top: 12px;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .sim-panel h3, .sim-panel p, .sim-panel label {
        color: #ffffff !important;
    }
    .sim-stat {
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 15px;
        padding: 16px;
        text-align: center;
    }
    .sim-stat .s-lbl {
        color: #e6f6f8;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .sim-stat .s-num {
        color: #ffd166;
        font-size: 1.85rem;
        font-weight: 800;
    }
    </style>
    <div class="hero-banner">
        <div class="hero-kicker">SENATI · INTELIGENCIA ARTIFICIAL · RETO INDIVIDUAL</div>
        <h1>PAVEMENT AI</h1>
        <div class="hero-subtitle">Predicción del deterioro del pavimento rígido mediante regresión lineal</div>
        <div class="highway-strip"></div>
        <div class="hero-meta">
            <span class="meta-pill">📍 HUÁNUCO · PERÚ</span>
            <span class="meta-pill">🤖 MACHINE LEARNING · LINEAR REGRESSION</span>
            <span class="meta-pill">⭐ EVALUACIÓN: 20 PUNTOS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Faltan el dataset o el modelo entrenado. Ejecuta `python data/generate_dataset.py` y `python model/train_model.py`.")
    st.stop()


@st.cache_data
def cargar_datos(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def cargar_modelo(path: Path) -> dict:
    return joblib.load(path)


def clasificar_pci(valor: float) -> str:
    if valor > 85:
        return "Bueno"
    if valor > 70:
        return "Satisfactorio"
    if valor > 55:
        return "Regular"
    if valor > 40:
        return "Pobre"
    if valor > 25:
        return "Muy pobre"
    if valor > 10:
        return "Grave"
    return "Fallado"


def predecir_pci_acotado(modelo, edad: float) -> float:
    val = float(modelo.predict(pd.DataFrame({"antiguedad": [float(edad)]}))[0])
    return float(np.clip(val, 0.0, 100.0))


df = cargar_datos(DATA_PATH)
artefacto = cargar_modelo(MODEL_PATH)
modelo = artefacto["modelo"]

if "vista_v3" not in st.session_state:
    st.session_state.vista_v3 = "01 · El reto"

secciones = [
    "01 · El reto",
    "02 · Datos",
    "03 · Prototipo",
    "04 · Desarrollo",
    "05 · Evaluación",
    "06 · Registro y Fotos",
]

nav_cols = st.columns(len(secciones))
for col, sec in zip(nav_cols, secciones):
    with col:
        es_activa = st.session_state.vista_v3 == sec
        if st.button(
            sec,
            key=f"btn_{sec}",
            type="primary" if es_activa else "secondary",
            use_container_width=True,
        ):
            st.session_state.vista_v3 = sec
            st.rerun()

vista = st.session_state.vista_v3

if vista == "01 · El reto":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Planteamiento del caso</div>', unsafe_allow_html=True)
    st.markdown("### 01 / El reto tecnológico")
    st.write(
        "Una municipalidad de Huánuco necesita analizar el deterioro de sus vías de concreto y planificar "
        "inspecciones futuras. Desarrolla una aplicación web que registre patologías y estime el índice de "
        "condición del pavimento (PCI) mediante regresión lineal."
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            '<div class="kpi-card"><div class="k-title">Objetivo</div>'
            '<p class="k-desc">Entrenar, evaluar e integrar un modelo predictivo de regresión lineal.</p></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            '<div class="kpi-card"><div class="k-title">Producto</div>'
            '<p class="k-desc">Dashboard interactivo, registro de patologías con fotos y simulador PCI.</p></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            '<div class="kpi-card"><div class="k-title">Modalidad</div>'
            '<p class="k-desc">Individual · nivel avanzado · calificación sobre 20 puntos.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Patologías del pavimento rígido")
    fila_a = st.columns(3)
    for i in range(3):
        nombre, d_path, desc_corta = PATOLOGIAS_SVG[i]
        with fila_a[i]:
            st.markdown(
                f'<div class="pat-box">'
                f'<svg viewBox="0 0 100 100" width="84" height="84">'
                f'<rect x="4" y="4" width="92" height="92" rx="10" fill="#e1eff5" stroke="#8fb3c7" stroke-width="2"/>'
                f'<path d="{d_path}" stroke="#1b3b6f" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
                f'</svg>'
                f'<span class="pat-name">{nombre}</span>'
                f'<span class="pat-sub">{desc_corta}</span></div>',
                unsafe_allow_html=True,
            )

    fila_b = st.columns(3)
    for i in range(3, 6):
        nombre, d_path, desc_corta = PATOLOGIAS_SVG[i]
        with fila_b[i - 3]:
            st.markdown(
                f'<div class="pat-box">'
                f'<svg viewBox="0 0 100 100" width="84" height="84">'
                f'<rect x="4" y="4" width="92" height="92" rx="10" fill="#e1eff5" stroke="#8fb3c7" stroke-width="2"/>'
                f'<path d="{d_path}" stroke="#1b3b6f" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
                f'</svg>'
                f'<span class="pat-name">{nombre}</span>'
                f'<span class="pat-sub">{desc_corta}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="callout-amber"><b>Nota metodológica:</b> El PCI observado debe provenir de una evaluación técnica; '
        'el modelo no reemplaza la metodología PCI ni una inspección estructural en campo.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif vista == "02 · Datos":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Fuente y estructura</div>', unsafe_allow_html=True)
    st.markdown("### 02 / Datos de entrenamiento")
    st.info("Conjunto de datos base **simulado con fines educativos** para tramos viales de referencia en Huánuco.")

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Registros totales", len(df))
    c_m2.metric("Tramos analizados", df["tramo_id"].nunique())
    c_m3.metric("Rango de antigüedad", f"{df['antiguedad'].min():.0f} – {df['antiguedad'].max():.0f} años")

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pavimento_huanuco_dataset.csv",
        mime="text/csv",
        type="primary",
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif vista == "03 · Prototipo":
    st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
    st.markdown("### 03 / Centro de análisis vial · PAVEMENT AI")
    st.write("Simulador respaldado por `LinearRegression` entrenado en Python (`antiguedad` → `pci_observado`).")

    escenarios = {
        "Personalizado": None,
        "Escenario 1 · Vía reciente (3 años + 2 años)": (3.0, 2.0),
        "Escenario 2 · Vía intermedia (6 años + 3 años)": (6.0, 3.0),
        "Escenario 3 · Extrapolación (10 años + 4 años)": (10.0, 4.0),
    }
    if "edad_v3" not in st.session_state:
        st.session_state.edad_v3 = 5.0
        st.session_state.fut_v3 = 3.0

    col_esc1, col_esc2 = st.columns([2.5, 1])
    with col_esc1:
        esc_sel = st.selectbox("Seleccionar escenario de prueba:", list(escenarios.keys()))
    with col_esc2:
        st.write("")
        st.write("")
        if st.button("Aplicar escenario", use_container_width=True) and escenarios[esc_sel] is not None:
            st.session_state.edad_v3, st.session_state.fut_v3 = escenarios[esc_sel]
            st.rerun()

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        edad_act = st.number_input("Antigüedad actual (años)", min_value=0.0, max_value=50.0, value=float(st.session_state.edad_v3), step=0.5)
    with col_in2:
        hor_fut = st.number_input("Horizonte futuro (años)", min_value=1.0, max_value=15.0, value=float(st.session_state.fut_v3), step=1.0)

    st.button("Calcular proyección PCI", type="primary", use_container_width=True)

    edad_total = float(edad_act + hor_fut)
    pci_act = predecir_pci_acotado(modelo, edad_act)
    pci_fut = predecir_pci_acotado(modelo, edad_total)
    cond_fut = clasificar_pci(pci_fut)

    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="sim-stat"><div class="s-lbl">PCI actual estimado</div><div class="s-num">{pci_act:.1f}</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="sim-stat"><div class="s-lbl">PCI futuro estimado</div><div class="s-num">{pci_fut:.1f}</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="sim-stat"><div class="s-lbl">Condición futura</div><div class="s-num">{cond_fut}</div></div>', unsafe_allow_html=True)

    pend = artefacto["pendiente"]
    inter = artefacto["intercepto"]
    signo = "+" if pend >= 0 else "-"
    st.markdown(f"**Ecuación del modelo:** `PCI = {inter:.2f} {signo} {abs(pend):.2f} × antigüedad`")

    if edad_total > artefacto["antiguedad_max"]:
        st.warning(
            f"Aviso de extrapolación: {edad_total:.1f} años supera el rango máximo del historial "
            f"({artefacto['antiguedad_max']:.1f} años). Interpretar con cautela."
        )

    eje_x = np.arange(1, max(16, int(np.ceil(edad_total)) + 2))
    eje_y = [predecir_pci_acotado(modelo, a) for a in eje_x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["antiguedad"], y=df["pci_observado"],
        mode="markers", name="Historial (simulado)",
        marker=dict(size=7, color="#80f0e6", opacity=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=eje_x, y=eje_y,
        mode="lines", name="Recta LinearRegression",
        line=dict(color="#ffd166", width=3, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[edad_total], y=[pci_fut],
        mode="markers", name="Predicción futura",
        marker=dict(size=13, color="#ff8a80", symbol="diamond"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.06)",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Antigüedad (años)", gridcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(title="PCI (0 – 100)", range=[0, 105], gridcolor="rgba(255,255,255,0.12)"),
        height=360,
        margin=dict(l=20, r=20, t=24, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif vista == "04 · Desarrollo":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Flujo técnico</div>', unsafe_allow_html=True)
    st.markdown("### 04 / Actividades de desarrollo")
    st.markdown(
        "1. Generación y limpieza del dataset con Pandas y NumPy (`data/generate_dataset.py` y `model/train_model.py`).\n"
        "2. División entrenamiento/prueba agrupada por tramo (`GroupShuffleSplit`) para evitar fuga de datos.\n"
        "3. Entrenamiento de `LinearRegression` tomando `antiguedad` como entrada y `pci_observado` como salida.\n"
        "4. Evaluación en prueba mediante **R²**, **MAE** y **RMSE**.\n"
        "5. Interfaz en Streamlit con validación, registro de patologías, evidencias fotográficas y exportación CSV."
    )
    st.code(
        "pip install -r requirements.txt\n"
        "python data/generate_dataset.py\n"
        "python model/train_model.py\n"
        "python -m streamlit run app/app.py",
        language="bash",
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif vista == "05 · Evaluación":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Desempeño en prueba</div>', unsafe_allow_html=True)
    st.markdown("### 05 / Métricas del modelo entrenado")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (Prueba)", f"{artefacto['r2']:.4f}")
    m2.metric("MAE (Prueba)", f"{artefacto['mae']:.2f}")
    m3.metric("RMSE (Prueba)", f"{artefacto['rmse']:.2f}")

    fig_bar = go.Figure(
        data=[go.Bar(
            x=df["tipo_falla"].value_counts().index,
            y=df["tipo_falla"].value_counts().values,
            marker_color="#16697a",
        )]
    )
    fig_bar.update_layout(
        title="Distribución de patologías en el dataset simulado",
        xaxis_title="Tipo de patología",
        yaxis_title="Cantidad de registros",
        height=330,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif vista == "06 · Registro y Fotos":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Inspección en campo</div>', unsafe_allow_html=True)
    st.markdown("### 06 / Registro de patologías y fotografías")

    with st.form("form_pat_v3"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tramo = st.text_input("Identificación de tramo", value="JR-28-DE-JULIO-T1")
            fecha_i = st.date_input("Fecha de inspección", value=date.today())
        with c2:
            ant_i = st.number_input("Antigüedad (años)", 0.0, 50.0, 5.0, 0.5)
            tran_i = st.number_input("Tránsito (veh/día)", 0, 50000, 1500, 50)
        with c3:
            pat_i = st.selectbox("Patología", [p[0] for p in PATOLOGIAS_SVG])
            sev_i = st.selectbox("Severidad", ["Baja", "Media", "Alta"])

        pci_i = st.number_input("PCI observado (0 – 100)", 0.0, 100.0, 75.0, 0.5)
        foto_i = st.file_uploader("Fotografía de evidencia (PNG/JPG)", type=["png", "jpg", "jpeg"])
        guardar = st.form_submit_button("Guardar registro", type="primary")

    if guardar:
        if not tramo.strip():
            st.error("Ingresa un identificador de tramo válido.")
        else:
            fname = ""
            if foto_i is not None:
                fname = f"{tramo.strip().replace(' ', '_')}_{Path(foto_i.name).name}"
                (UPLOADS_DIR / fname).write_bytes(foto_i.getbuffer())

            nueva_fila = pd.DataFrame([{
                "tramo_id": tramo.strip(),
                "fecha": fecha_i.strftime("%Y-%m-%d"),
                "antiguedad": float(ant_i),
                "transito": int(tran_i),
                "tipo_falla": pat_i,
                "severidad": sev_i,
                "pci_observado": float(pci_i),
                "foto": fname,
            }])
            df_reg = pd.concat([pd.read_csv(PATOLOGIAS_PATH), nueva_fila], ignore_index=True) if PATOLOGIAS_PATH.exists() else nueva_fila
            df_reg.to_csv(PATOLOGIAS_PATH, index=False, encoding="utf-8")
            st.success("Registro guardado correctamente.")

    if PATOLOGIAS_PATH.exists():
        df_show = pd.read_csv(PATOLOGIAS_PATH)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar registros a CSV",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name="patologias_registradas.csv",
            mime="text/csv",
        )
        con_foto = df_show[df_show["foto"].fillna("") != ""]
        if not con_foto.empty:
            st.markdown("#### Fotografías cargadas")
            for _, row in con_foto.tail(4).iterrows():
                img_p = UPLOADS_DIR / str(row["foto"])
                if img_p.exists():
                    st.image(str(img_p), caption=f"{row['tramo_id']} · {row['tipo_falla']} ({row['fecha']})", width=340)

    st.markdown("</div>", unsafe_allow_html=True)
