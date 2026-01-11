import streamlit as st
import numpy as np
import pandas as pd
from MOMENTOS import LIMomento
from CORTANTE import LICortante
from REACCIONES import LIREACCION
from CARGACARRIL import CargarCarril
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
import tempfile
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN Y ESTILOS
# ============================================================
st.set_page_config(
    layout="centered", 
    page_title="Líneas de Influencia - Puentes",
    page_icon="https://cdn.pixabay.com/photo/2013/07/12/18/35/alien-153542_1280.png",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2.6rem;
        font-weight: 900;
        color: #ffffff;
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, #141e30, #243b55);
        margin: 2.5rem 0 1.7rem 0;
        border-radius: 14px;
        letter-spacing: 1px;
        box-shadow: 
            0 10px 25px rgba(0, 0, 0, 0.8),
            0 0 20px rgba(52, 152, 219, 0.35);
        border-left: 6px solid #00c6ff;
        text-shadow: 0 0 8px rgba(0, 198, 255, 0.4);
    }

.info-box {
    display: inline-block;
    width: fit-content;
    max-width: 100%;
    padding: 0.9rem 1.4rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: #eaf4ff;
    font-size: 1.1rem;
    box-shadow: 
        0 6px 15px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05);
    border-left: 5px solid #4aa3ff;
    margin: 0.5rem 0 1rem 0;
    box-sizing: border-box;
}

    .equation-box {
        background-color: #fffef0;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #ffd700;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        font-size: 1.1rem;
        color: #2c3e50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
    }
            
    .author-box {
    text-align: center;
    margin-top: -10px;
    margin-bottom: 10px;
    color: #eaf4ff;
    font-size: 1.1rem;
    opacity: 0.9;
}

.social-links {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    margin-bottom: 1rem;
}

.social-links a {
    text-decoration: none;
    color: #4aa3ff;
    font-weight: 600;
}

.social-links a:hover {
    text-decoration: underline;
}

.app-description {
    text-align: center;
    font-size: 1.1rem;
    opacity: 0.85;
    margin-bottom: 1.5rem;
}
                    
</style>
""", unsafe_allow_html=True)

def generar_pdf(figs, textos):
    """Genera un PDF con las figuras y textos proporcionados"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    for texto in textos:
        story.append(Paragraph(texto, styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))
    
    for fig in figs:
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        story.append(Image(img_path, width=16 * cm, height=9 * cm))
        story.append(Spacer(1, 1 * cm))
    
    doc.build(story)
    return temp_file.name

def plot_hl93_sobre_linea_influencia(VInfo, EjeCentral, EjdeDelantero, EjePosterior, 
                                      LiCentral, LiDelantero, LiPosterior,
                                      CargaCentral, CargaDelantera, CargaPosterior, 
                                      Step, LI_Geom):
    """Grafica la línea de influencia con las cargas HL-93 posicionadas"""
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dibujar línea de influencia
    ax.plot(VInfo[:, 0], VInfo[:, 1], 'b-', linewidth=2, label='Línea de Influencia')
    
    # Rellenar áreas
    plt.fill_between(VInfo[:, 0], VInfo[:, 1], 0, where=(VInfo[:, 1] < 0), 
                     color='lightcoral', alpha=0.3, label='Área Negativa')
    plt.fill_between(VInfo[:, 0], VInfo[:, 1], 0, where=(VInfo[:, 1] >= 0), 
                     color='skyblue', alpha=0.3, label='Área Positiva')
    
    # Dibujar línea base del puente (nodos)
    x_coords = np.hstack((LI_Geom[:, 0], LI_Geom[-1, 1]))
    y_coords = np.zeros(len(x_coords))
    ax.plot(x_coords, y_coords, color='#000000', linewidth=1.5)
    
    # Dibujar nodos (apoyos)
    markers = ['s'] + ['^'] * (len(x_coords) - 1)
    colors_nodos = ['#000000'] * len(x_coords)
    for x, y, marker, color in zip(x_coords, y_coords, markers, colors_nodos):
        ax.scatter(x, y, color=color, marker=marker, s=60, zorder=5)
    
    # Recalcular las ordenadas directamente desde VInfo para cada eje
    # Buscar los índices más cercanos a cada posición de eje
    idx_central = np.argmin(np.abs(VInfo[:, 0] - EjeCentral))
    idx_delantero = np.argmin(np.abs(VInfo[:, 0] - EjdeDelantero))
    idx_posterior = np.argmin(np.abs(VInfo[:, 0] - EjePosterior))
    
    # Obtener las ordenadas reales desde VInfo
    Li_central_real = VInfo[idx_central, 1]
    Li_delantero_real = VInfo[idx_delantero, 1]
    Li_posterior_real = VInfo[idx_posterior, 1]
    
    # Marcar el punto de máximo momento
    max_index = np.argmax(np.abs(VInfo[:, 1]))
    max_value = VInfo[max_index, 1]
    max_x = VInfo[max_index, 0]
    ax.plot(max_x, max_value, 'ro', markersize=7, zorder=5)
    ax.plot([max_x, max_x], [max_value, 0], color='red', linestyle='--', linewidth=1)
    
    # Dibujar cargas HL-93 con ordenadas recalculadas
    cargas = [
        (EjeCentral, Li_central_real, CargaCentral, 'Central', 'red'),
        (EjdeDelantero, Li_delantero_real, CargaDelantera, 'Delantera', 'green'),
        (EjePosterior, Li_posterior_real, CargaPosterior, 'Posterior', 'orange')
    ]
    
    # Calcular rango de valores para posicionamiento inteligente
    y_range = VInfo[:, 1].max() - VInfo[:, 1].min()
    
    for idx, (eje_x, ordenada, carga, nombre, color) in enumerate(cargas):
        # Línea vertical desde la ordenada hasta el eje x
        ax.plot([eje_x, eje_x], [0, ordenada], color=color, linestyle='--', linewidth=1.5)
        
        # Punto en la línea de influencia
        ax.plot(eje_x, ordenada, 'o', color=color, markersize=10, 
                label=f'{nombre}: {carga:.2f} ton', zorder=4)
        
        # Flecha de carga (ajustar según si es positivo o negativo)
        if ordenada > 0:
            flecha_inicio = ordenada + abs(y_range * 0.12)
            flecha_long = -abs(y_range * 0.08)
        else:
            flecha_inicio = ordenada - abs(y_range * 0.12)
            flecha_long = abs(y_range * 0.08)
        
        ax.arrow(eje_x, flecha_inicio, 0, flecha_long, 
                head_width=0.5, head_length=abs(flecha_long)*0.3, 
                fc=color, ec=color, linewidth=2, zorder=3)
        
        # Anotación con valor de ordenada - posicionamiento inteligente
        # Solo etiquetar si NO es el punto máximo (para evitar duplicados)
        if abs(eje_x - max_x) > 0.1:  # Si no está cerca del máximo
            x_center = (VInfo[:, 0].min() + VInfo[:, 0].max()) / 2
            
            if eje_x < x_center:
                offset_x = -1.8
                ha = 'right'
            else:
                offset_x = 1.8
                ha = 'left'
            
            # Offset vertical más pequeño y pegado a la línea
            offset_y = y_range * 0.02
            
            ax.annotate(f'Li={ordenada:.4f}', 
                       xy=(eje_x, ordenada), 
                       xytext=(eje_x + offset_x, ordenada + offset_y),
                       fontsize=9,
                       ha=ha,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6, edgecolor=color, linewidth=1.5),
                       zorder=6)
        else:
            # Si es el máximo, etiquetarlo especialmente
            ax.annotate(f'Li={ordenada:.4f}\nMáx: {max_value:.3f}', 
                       xy=(eje_x, ordenada), 
                       xytext=(eje_x + 1.2, ordenada + y_range * 0.05),
                       fontsize=9,
                       ha='left',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2),
                       arrowprops=dict(facecolor='black', arrowstyle='-', linewidth=1),
                       zorder=7)
    
    # Centrar y ajustar límites de la gráfica
    x_min, x_max = VInfo[:, 0].min(), VInfo[:, 0].max()
    x_margin = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    y_min, y_max = VInfo[:, 1].min(), VInfo[:, 1].max()
    y_range = y_max - y_min
    y_margin = y_range * 0.2
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.set_xlabel('Distancia del puente (m)', fontsize=12)
    ax.set_ylabel('Valor en la línea de influencia', fontsize=12)
    ax.set_title('LÍNEA DE INFLUENCIA DE MOMENTOS CON CARGA HL-93', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.axhline(y=0, color='k', linewidth=0.8)
    
    return fig

def mostrar_ecuacion_momento(Li_central, Li_delantero, Li_posterior, 
                            P_central, P_delantera, P_posterior, M_total):
    """Muestra las ecuaciones de momento estilo handcalcs con LaTeX"""
    st.markdown("#### 📝 Procedimiento de Cálculo")
    
    st.markdown("**Paso 1: Identificar ordenadas de la línea de influencia**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(f"L_{{i,central}} = {Li_central:.4f}")
    with col2:
        st.latex(f"L_{{i,delantero}} = {Li_delantero:.4f}")
    with col3:
        st.latex(f"L_{{i,posterior}} = {Li_posterior:.4f}")
    
    st.markdown("**Paso 2: Cargas aplicadas (HL-93)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(f"P_{{central}} = {P_central:.2f} \\text{{ ton}}")
    with col2:
        st.latex(f"P_{{delantera}} = {P_delantera:.2f} \\text{{ ton}}")
    with col3:
        st.latex(f"P_{{posterior}} = {P_posterior:.2f} \\text{{ ton}}")
    
    st.markdown("**Paso 3: Cálculo del momento por superposición**")
    
    st.latex(r"M = \sum (L_i \times P)")
    
    st.latex(f"M = L_{{i,central}} \\times P_{{central}} + L_{{i,delantero}} \\times P_{{delantera}} + L_{{i,posterior}} \\times P_{{posterior}}")
    
    st.latex(f"M = {Li_central:.4f} \\times {P_central:.2f} + {Li_delantero:.4f} \\times {P_delantera:.2f} + {Li_posterior:.4f} \\times {P_posterior:.2f}")
    
    st.latex(f"M = {Li_central * P_central:.4f} + {Li_delantero * P_delantera:.4f} + {Li_posterior * P_posterior:.4f}")
    
    st.markdown(f'<div class="result-box"><b>✅ Resultado Final:</b></div>', unsafe_allow_html=True)
    st.latex(f"M = {M_total:.4f} \\text{{ ton}} \\cdot \\text{{m}}")

def mostrar_ecuacion_carril(areas, factor_carga, M_total, tramos_con_carga):
    """Muestra las ecuaciones para carga de carril con LaTeX"""
    st.markdown("#### 📝 Procedimiento de Cálculo")
    
    st.markdown("**Paso 1: Áreas bajo la línea de influencia**")
    st.latex(r"A_{total} = \sum (A_{positiva} + |A_{negativa}|)")
    
    st.markdown("**Paso 2: Cálculo por tramo**")
    
    area_total_sum = 0
    for i, (area_neg, area_pos) in enumerate(areas):
        if tramos_con_carga[i]:  # Solo mostrar si el tramo tiene carga
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Tramo {i+1}:** ✅")
            with col2:
                st.latex(f"A_{{neg}} = {area_neg:.4f}")
            with col3:
                st.latex(f"A_{{pos}} = {area_pos:.4f}")
            
            area_tramo = area_neg + area_pos
            st.latex(f"A_{{total,{i+1}}} = {area_neg:.4f} + {area_pos:.4f} = {area_tramo:.4f}")
            area_total_sum += area_tramo
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Tramo {i+1}:** ❌ (sin carga)")
    
    st.markdown("**Paso 3: Sumar áreas de tramos con carga**")
    st.latex(f"A_{{total}} = {area_total_sum:.4f}")
    
    st.markdown("**Paso 4: Aplicar intensidad de carga uniforme**")
    
    st.latex(f"w = {factor_carga:.3f} \\text{{ ton/m}}")
    
    st.latex(r"M = A_{total} \times w")
    
    st.latex(f"M = {area_total_sum:.4f} \\times {factor_carga:.3f}")
    
    st.markdown(f'<div class="result-box"><b>✅ Resultado Final:</b></div>', unsafe_allow_html=True)
    st.latex(f"M = {M_total:.4f} \\text{{ ton}} \\cdot \\text{{m}}")

# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================
st.markdown('<h1 class="main-header">TJGO Bridge Tools</h1>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="author-box">
        <strong> Ñol Iván Juan de Dios Rojas</strong><br>
        <span>  Ingeniería Civil Estructural   | Puentes | Automatización </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<div class="social-links">
    <a href="https://github.com/TJGO-JDD" target="_blank">
        <i class="fab fa-github"></i><span>GitHub</span>
    </a>
    <a href="https://www.linkedin.com/in/%C3%B1ol-ivan-juan-de-dios-rojas-600b36273/" target="_blank">
        <i class="fab fa-linkedin"></i><span>LinkedIn</span>
    </a>
    <a href="https://api.whatsapp.com/send/?phone=51917726087&text=Deseo+informaci%C3%B3n+sobre%21+....&type=phone_number&app_absent=0" target="_blank">
        <i class="fab fa-whatsapp"></i><span>WhatsApp</span>
    </a>
    <a href="https://www.instagram.com/ivan_jdd_tjgo/" target="_blank">
        <i class="fab fa-instagram"></i><span>Instagram</span>
    </a>
    <a href="https://www.youtube.com/@TJGO-JDD" target="_blank">
        <i class="fab fa-youtube"></i><span>YouTube</span>
    </a>
    <a href="https://x.com/ivanjuandedios1" target="_blank">
        <i class="fab fa-x-twitter"></i><span>X</span>
    </a>
    <a href="https://www.tiktok.com/@ivanjddhood" target="_blank">
        <i class="fab fa-tiktok"></i><span>TikTok</span>
    </a>
</div>
""", unsafe_allow_html=True)



st.markdown(
    """
    <div class="app-description">
        Plataforma para el cálculo automático de líneas de influencia, momentos flectores, fuerzas cortantes y cargas de carril en puentes.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ============================================================
# SIDEBAR - PARÁMETROS GENERALES
# ============================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/TJGO-JDD/TJGO-JDD.github.io/refs/heads/main/assets/img/LOGO2.png", 
             use_container_width=True)
    st.markdown("## ÑOL IVAN JUAN DE DIOS ROJAS | TJGO 2026")
    st.markdown("---")
    
    Step = st.number_input(" Step de análisis", value=0.01, format="%.4f", 
                          help="Incremento para el cálculo de la línea de influencia")
    
    st.markdown("---")
    st.markdown("### 📊 Información del Proyecto")
    proyecto = st.text_input("Nombre del proyecto", "Puente Principal")
    analista = st.text_input("Analista", "Ing. Civil")
    fecha = st.date_input("Fecha de análisis")

# ============================================================
# 1️⃣ GEOMETRÍA DEL PUENTE
# ============================================================
st.markdown('<h2 class="section-header">📐 Geometría del Puente</h2>', 
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    st.info("📌 Define la geometría del puente mediante tramos. Cada fila representa un tramo con sus propiedades.")

with col_info2:
    st.metric("Número de tramos", 
              len(st.session_state.get("geom_df", pd.DataFrame({"xi": [0]}))))


if "geom_df" not in st.session_state:
    st.session_state.geom_df = pd.DataFrame({
        "xi": [0, 12.75],
        "xf": [12.75, 28.25],
        "E": [1, 1],
        "I": [1, 1]
    })

geom_df = st.data_editor(
    st.session_state.geom_df, 
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "xi": st.column_config.NumberColumn("Inicio (m)", format="%.2f"),
        "xf": st.column_config.NumberColumn("Fin (m)", format="%.2f"),
        "E": st.column_config.NumberColumn("Módulo E", format="%.2f"),
        "I": st.column_config.NumberColumn("Inercia I", format="%.4f"),
    }
)
st.session_state.geom_df = geom_df
LI_Geom = geom_df.to_numpy()

# Mostrar longitud total
longitud_total = LI_Geom[-1, 1]
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <div class="info-box">
            <b> Longitud total del puente:</b> {longitud_total:.2f} m
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# ============================================================
# PUNTO DE ANÁLISIS
# ============================================================
st.markdown('<span class="section-badge">🎯 Punto de Análisis</span>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    Panalisis = st.slider(
        "Selecciona la posición X para análisis (m)",
        min_value=0.0,
        max_value=float(longitud_total),
        value=float(LI_Geom[0, 1] * 0.4),
        step=0.01
    )
    st.markdown(f'<div class="info-box" style="text-align: center;"><b>Punto seleccionado:</b> X = {Panalisis:.2f} m</div>', 
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ============================================================
# 3️⃣ LÍNEA DE INFLUENCIA DE MOMENTOS
# ============================================================
st.markdown('<h2 class="section-header">📊 Línea de Influencia de Momentos</h2>', 
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

LIMom = LIMomento(LI_Geom, Panalisis, Step)
fig_mom = LIMom.plot()

col_left, col_graph, col_right = st.columns([0.5, 3, 0.5])

with col_graph:
    st.pyplot(fig_mom, use_container_width=False)

st.markdown("### 📈 Métricas Principales")
col1, col2, col3 = st.columns(3)
col1.metric("🔺 Valor Máximo", f"{LIMom.MaxValue():.4f}")
col2.metric("➕ Área Positiva", f"{LIMom.AreaPositiva():.4f}")
col3.metric("➖ Área Negativa", f"{LIMom.AreaNegativa():.4f}")

Areas = LIMom.Atramos()
VInfo = LIMom.calculate_VInfo()

st.markdown("### 📋 Áreas por Tramo")
areas_df = pd.DataFrame(Areas, columns=["Área Negativa", "Área Positiva"])
areas_df.index = [f"Tramo {i+1}" for i in range(len(areas_df))]
st.dataframe(areas_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

# ============================================================
# BUSCAR ORDENADA EN CUALQUIER PUNTO X - MOMENTOS
# ============================================================
st.markdown("### 🔍 Consultar Ordenada en Punto Específico")

st.markdown("""
<div class="info-box" style="text-align: center;">
    <strong>ℹ️ Buscar ordenada:
     Ingresa cualquier posición X dentro del puente para obtener 
    la ordenada de la línea de influencia de momentos en ese punto.
</div>
""", unsafe_allow_html=True)

col_buscar1, col_buscar2, col_buscar3 = st.columns([1, 2, 1])

with col_buscar2:
    x_consulta = st.number_input(
        "📍 Posición X a consultar (m)",
        min_value=0.0,
        max_value=float(longitud_total),
        value=float(longitud_total / 2),
        step=0.1,
        key="x_consulta_momento"
    )
    
    if st.button("🔎 Buscar Ordenada de Momento", use_container_width=True, key="btn_buscar_momento"):
        idx_consulta = np.argmin(np.abs(VInfo[:, 0] - x_consulta))
        x_real = VInfo[idx_consulta, 0]
        ordenada_consulta = VInfo[idx_consulta, 1]
        
        st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**📍 Posición consultada:** X = {x_consulta:.4f} m")
        st.markdown(f"**📏 Posición real más cercana:** X = {x_real:.4f} m")
        st.markdown(f"**📊 Ordenada de LI Momento:** {ordenada_consulta:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.latex(f"L_i(x={x_real:.4f}) = {ordenada_consulta:.6f}")
        
        if ordenada_consulta > 0:
            st.success(f"✅ La ordenada es **positiva** ({ordenada_consulta:.6f})")
        elif ordenada_consulta < 0:
            st.warning(f"⚠️ La ordenada es **negativa** ({ordenada_consulta:.6f})")
        else:
            st.info(f"ℹ️ La ordenada es **cero** (punto de cambio de signo)")

st.markdown("---")


# ============================================================
# 4️⃣ MOMENTO POR CARGA HL-93
# ============================================================

st.markdown("## 📊 Análisis de Momentos")

st.markdown("### 🚛 Configuración de Carga HL-93 para Momentos")

st.markdown("""
<div class="info-box">
    <strong>ℹ️ Carga HL-93 (AASHTO):</strong> Configuración de camión de diseño con tres ejes para análisis de momentos.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**⚙️ Separación de Ejes - Momentos**")
    separacion_delantero_mom = st.number_input(
        "Separación eje delantero (m)", 
        value=4.3, 
        step=0.1,
        key="sep_delantero_mom",
        help="Distancia entre eje central y eje delantero (hacia atrás)"
    )
    separacion_posterior_mom = st.number_input(
        "Separación eje posterior (m)", 
        value=4.3, 
        step=0.1,
        key="sep_posterior_mom",
        help="Distancia entre eje central y eje posterior (hacia adelante)"
    )

with col2:
    st.markdown("**🚛 Cargas de los Ejes - Momentos**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        CargaCentral_Mom = st.number_input("🔵 Central (ton)", value=14.54, step=0.1, key="carga_central_mom")
    with col_b:
        CargaDelantera_Mom = st.number_input("🟢 Delantera (ton)", value=14.54, step=0.1, key="carga_delantero_mom")
    with col_c:
        CargaPosterior_Mom = st.number_input("🟡 Posterior (ton)", value=3.63, step=0.1, key="carga_posterior_mom")

EjeCentral_Mom = Panalisis
EjdeDelantero_Mom = Panalisis - separacion_delantero_mom
EjePosterior_Mom = Panalisis + separacion_posterior_mom



try:
    # Recalcular las ordenadas directamente desde VInfo
    idx_central = np.argmin(np.abs(VInfo[:, 0] - EjeCentral_Mom))
    idx_delantero = np.argmin(np.abs(VInfo[:, 0] - EjdeDelantero_Mom))
    idx_posterior = np.argmin(np.abs(VInfo[:, 0] - EjePosterior_Mom))
    
    # Obtener las ordenadas reales desde VInfo
    LiCentral = VInfo[idx_central, 1]
    LiDelantero = VInfo[idx_delantero, 1]
    LiPosterior = VInfo[idx_posterior, 1]
    
    Mtotal = (
        LiCentral * CargaCentral_Mom +
        LiDelantero * CargaDelantera_Mom +
        LiPosterior * CargaPosterior_Mom
    )
    
    # Graficar la línea de influencia con las cargas HL-93
    st.markdown("### 📊 Visualización de Cargas sobre Línea de Influencia")
    fig_hl93 = plot_hl93_sobre_linea_influencia(
        VInfo, EjeCentral_Mom, EjdeDelantero_Mom, EjePosterior_Mom,
        LiCentral, LiDelantero, LiPosterior,
        CargaCentral_Mom, CargaDelantera_Mom, CargaPosterior_Mom, Step, LI_Geom
    )
    
    col_left_hl, col_graph_hl, col_right_hl = st.columns([0.5, 3, 0.5])
    with col_graph_hl:
        st.pyplot(fig_hl93, use_container_width=False)
    
    # Mostrar ecuaciones estilo handcalcs
    with st.expander("🧮 Ver Cálculos Detallados", expanded=True):
        mostrar_ecuacion_momento(
            LiCentral, LiDelantero, LiPosterior,
            CargaCentral_Mom, CargaDelantera_Mom, CargaPosterior_Mom,
            Mtotal
        )
    
except Exception as e:
    st.error("⚠️ Alguno de los ejes está fuera del dominio del puente")
    st.exception(e)

# ============================================================
# MOMENTO POR CARGA DE CARRIL
# ============================================================
st.markdown("### 🛣️ Momento por Carga de Carril")

st.markdown("""
<div class="info-box">
    <strong>ℹ️ Carga de carril:</strong> Carga uniforme distribuida que simula el peso del tráfico. 
    Selecciona los tramos donde aplicar la carga.
</div>
""", unsafe_allow_html=True)

st.markdown("### 🎚️ Selección de Tramos con Carga")
tramos_con_carga = []
cols = st.columns(len(Areas))
for i, col in enumerate(cols):
    with col:
        tramos_con_carga.append(
            st.checkbox(f"Tramo {i+1}", value=True, key=f"tramo_{i}")
        )

num_fle = st.number_input("🔢 Número de flechas para visualización", 
                          value=4, step=1, min_value=2, max_value=10)

Carga_Carril = CargarCarril(VInfo, LI_Geom, tramos_con_carga, num_fle)
fig_carril = Carga_Carril.graficar()

col_left2, col_graph2, col_right2 = st.columns([0.5, 3, 0.5])

with col_graph2:
    st.pyplot(fig_carril, use_container_width=False)

st.markdown("### 📊 Resultados")
factor_carga = 0.952
MomentoCargaCarril = np.zeros(len(tramos_con_carga))

for idx, tiene_carga in enumerate(tramos_con_carga):
    if tiene_carga:
        MomentoCargaCarril[idx] = (Areas[idx, 0] + Areas[idx, 1]) * factor_carga
    else:
        MomentoCargaCarril[idx] = 0

M_carril_total = np.sum(MomentoCargaCarril)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.metric("🎯 Momento Total", f"{M_carril_total:.4f} ton·m")

# Mostrar cálculos detallados
with st.expander("🧮 Ver Cálculos Detallados", expanded=False):
    mostrar_ecuacion_carril(Areas, factor_carga, M_carril_total, tramos_con_carga)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ============================================================
# 6️⃣ ANÁLISIS DE CORTANTES
# ============================================================
st.markdown('<h2 class="section-header">✂️ Análisis de Cortantes</h2>', 
            unsafe_allow_html=True)

st.markdown("### 📊 Línea de Influencia de Cortantes")

LICor = LICortante(LI_Geom, Panalisis, Step)
fig_cor = LICor.plot()

col_left3, col_graph3, col_right3 = st.columns([0.5, 3, 0.5])
with col_graph3:
    st.pyplot(fig_cor, use_container_width=False)

# Obtener VInfo para cortantes (usar el atributo VInfo directamente)
VInfo_Cortante = LICor.VInfo

st.markdown("### 📈 Métricas de Cortante")
col1, col2, col3 = st.columns(3)
col1.metric("🔺 Valor Máximo", f"{LICor.MaxValue():.4f}")
col2.metric("➕ Área Positiva", f"{LICor.AreaPositiva():.4f}")
col3.metric("➖ Área Negativa", f"{LICor.AreaNegativa():.4f}")

# ============================================================
# CORTANTE POR CARGA HL-93
# ============================================================
st.markdown("### 🚛 Configuración de Carga HL-93 para Cortantes")

# Switch para seleccionar tipo de cortante
col_switch, col_info = st.columns([1, 2])
with col_switch:
    tipo_cortante_hl93 = st.radio(
        "Tipo de Cortante a calcular",
        options=["Positivo", "Negativo"],
        horizontal=True,
        key="tipo_cortante_hl93"
    )
with col_info:
    st.info(f"📍 Se posicionará el camión en el máximo cortante {tipo_cortante_hl93.lower()}")

st.info(f"ℹ️ Carga HL-93 para Cortante: Configuración independiente del camión para análisis de cortantes.")


col1, col2 = st.columns(2)
with col1:
    st.markdown("**⚙️ Separación de Ejes - Cortantes**")
    separacion_delantero_cor = st.number_input(
        "Separación eje delantero (m)", 
        value=4.3, 
        step=0.1,
        key="sep_delantero_cor",
        help="Distancia entre eje central y eje delantero (hacia atrás)"
    )
    separacion_posterior_cor = st.number_input(
        "Separación eje posterior (m)", 
        value=4.3, 
        step=0.1,
        key="sep_posterior_cor",
        help="Distancia entre eje central y eje posterior (hacia adelante)"
    )

with col2:
    st.markdown("**🚛 Cargas de los Ejes - Cortantes**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        CargaCentral_Cor = st.number_input("🔵 Central (ton)", value=14.54, step=0.1, key="carga_central_cor")
    with col_b:
        CargaDelantera_Cor = st.number_input("🟢 Delantera (ton)", value=14.54, step=0.1, key="carga_delantero_cor")
    with col_c:
        CargaPosterior_Cor = st.number_input("🟡 Posterior (ton)", value=3.63, step=0.1, key="carga_posterior_cor")

# Encontrar el punto de máximo cortante según el tipo seleccionado
if tipo_cortante_hl93 == "Positivo":
    # Buscar máximo positivo
    idx_max_cortante = np.argmax(VInfo_Cortante[:, 1])
else:
    # Buscar máximo negativo (mínimo)
    idx_max_cortante = np.argmin(VInfo_Cortante[:, 1])

Punto_max_cortante = VInfo_Cortante[idx_max_cortante, 0]

# Posicionar el camión en el punto de máximo cortante
EjeCentral_Cor = Punto_max_cortante
EjdeDelantero_Cor = Punto_max_cortante - separacion_delantero_cor
EjePosterior_Cor = Punto_max_cortante + separacion_posterior_cor

st.markdown(f"**📍 Posición del camión:** Eje central en X = {Punto_max_cortante:.2f} m (máximo cortante {tipo_cortante_hl93.lower()})")

st.markdown("#### 📊 Cálculo de Cortante HL-93")

try:
    # Recalcular las ordenadas directamente desde VInfo_Cortante
    idx_central_cor = np.argmin(np.abs(VInfo_Cortante[:, 0] - EjeCentral_Cor))
    idx_delantero_cor = np.argmin(np.abs(VInfo_Cortante[:, 0] - EjdeDelantero_Cor))
    idx_posterior_cor = np.argmin(np.abs(VInfo_Cortante[:, 0] - EjePosterior_Cor))
    
    # Obtener las ordenadas reales desde VInfo_Cortante
    LiCentral_Cor_raw = VInfo_Cortante[idx_central_cor, 1]
    LiDelantero_Cor_raw = VInfo_Cortante[idx_delantero_cor, 1]
    LiPosterior_Cor_raw = VInfo_Cortante[idx_posterior_cor, 1]
    
    # Filtrar ordenadas según el tipo de cortante seleccionado
    if tipo_cortante_hl93 == "Positivo":
        # Solo considerar ordenadas positivas, las negativas son 0
        LiCentral_Cor = LiCentral_Cor_raw if LiCentral_Cor_raw > 0 else 0
        LiDelantero_Cor = LiDelantero_Cor_raw if LiDelantero_Cor_raw > 0 else 0
        LiPosterior_Cor = LiPosterior_Cor_raw if LiPosterior_Cor_raw > 0 else 0
    else:  # Negativo
        # Solo considerar ordenadas negativas, las positivas son 0
        LiCentral_Cor = LiCentral_Cor_raw if LiCentral_Cor_raw < 0 else 0
        LiDelantero_Cor = LiDelantero_Cor_raw if LiDelantero_Cor_raw < 0 else 0
        LiPosterior_Cor = LiPosterior_Cor_raw if LiPosterior_Cor_raw < 0 else 0
    
    Vtotal = (
        LiCentral_Cor * CargaCentral_Cor +
        LiDelantero_Cor * CargaDelantera_Cor +
        LiPosterior_Cor * CargaPosterior_Cor
    )
    
    # Graficar la línea de influencia con las cargas HL-93
    st.markdown("#### 📊 Visualización de Cargas sobre Línea de Influencia de Cortante")
    fig_hl93_cor = plot_hl93_sobre_linea_influencia(
        VInfo_Cortante, EjeCentral_Cor, EjdeDelantero_Cor, EjePosterior_Cor,
        LiCentral_Cor, LiDelantero_Cor, LiPosterior_Cor,
        CargaCentral_Cor, CargaDelantera_Cor, CargaPosterior_Cor, Step, LI_Geom
    )
    
    col_left_hl_cor, col_graph_hl_cor, col_right_hl_cor = st.columns([0.5, 3, 0.5])
    with col_graph_hl_cor:
        st.pyplot(fig_hl93_cor, use_container_width=False)
    
    # Mostrar ecuaciones estilo handcalcs para cortante
    with st.expander("🧮 Ver Cálculos Detallados de Cortante HL-93", expanded=True):
        st.markdown(f"#### 📝 Procedimiento de Cálculo - Cortante {tipo_cortante_hl93}")
        
        st.markdown(f"**Paso 1: Identificar ordenadas de la línea de influencia de cortante {tipo_cortante_hl93.lower()}**")
        st.info(f"ℹ️ Solo se consideran ordenadas {tipo_cortante_hl93.lower()}s. Las ordenadas del signo opuesto se toman como 0.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(f"L_{{i,central}} = {LiCentral_Cor:.4f}")
            if LiCentral_Cor == 0 and LiCentral_Cor_raw != 0:
                st.caption(f"(valor real: {LiCentral_Cor_raw:.4f}, ignorado)")
        with col2:
            st.latex(f"L_{{i,delantero}} = {LiDelantero_Cor:.4f}")
            if LiDelantero_Cor == 0 and LiDelantero_Cor_raw != 0:
                st.caption(f"(valor real: {LiDelantero_Cor_raw:.4f}, ignorado)")
        with col3:
            st.latex(f"L_{{i,posterior}} = {LiPosterior_Cor:.4f}")
            if LiPosterior_Cor == 0 and LiPosterior_Cor_raw != 0:
                st.caption(f"(valor real: {LiPosterior_Cor_raw:.4f}, ignorado)")
        
        st.markdown("**Paso 2: Cargas aplicadas (HL-93)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(f"P_{{central}} = {CargaCentral_Cor:.2f} \\text{{ ton}}")
        with col2:
            st.latex(f"P_{{delantera}} = {CargaDelantera_Cor:.2f} \\text{{ ton}}")
        with col3:
            st.latex(f"P_{{posterior}} = {CargaPosterior_Cor:.2f} \\text{{ ton}}")
        
        st.markdown("**Paso 3: Cálculo del cortante por superposición**")
        
        st.latex(r"V = \sum (L_i \times P)")
        
        st.latex(f"V = L_{{i,central}} \\times P_{{central}} + L_{{i,delantero}} \\times P_{{delantera}} + L_{{i,posterior}} \\times P_{{posterior}}")
        
        st.latex(f"V = {LiCentral_Cor:.4f} \\times {CargaCentral_Cor:.2f} + {LiDelantero_Cor:.4f} \\times {CargaDelantera_Cor:.2f} + {LiPosterior_Cor:.4f} \\times {CargaPosterior_Cor:.2f}")
        
        st.latex(f"V = {LiCentral_Cor * CargaCentral_Cor:.4f} + {LiDelantero_Cor * CargaDelantera_Cor:.4f} + {LiPosterior_Cor * CargaPosterior_Cor:.4f}")
        
        st.markdown(f'<div class="result-box"><b>✅ Resultado Final - Cortante {tipo_cortante_hl93}:</b></div>', unsafe_allow_html=True)
        st.latex(f"V = {Vtotal:.4f} \\text{{ ton}}")
    
except Exception as e:
    st.error("⚠️ Alguno de los ejes está fuera del dominio del puente para cortante")
    st.exception(e)

# ============================================================
# CORTANTE POR CARGA DE CARRIL
# ============================================================
st.markdown("### 🛣️ Cortante por Carga de Carril")

st.info(f"ℹ️ Cortante por carril: Selecciona si deseas calcular cortante positivo o negativo. El cortante positivo usa áreas positivas, el negativo usa áreas negativas.")



# Switch para seleccionar tipo de cortante
col_switch, col_empty = st.columns([1, 3])
with col_switch:
    tipo_cortante = st.radio(
        "Tipo de Cortante",
        options=["Positivo", "Negativo"],
        horizontal=True,
        key="tipo_cortante"
    )

st.markdown(f"**Calculando Cortante {tipo_cortante}** {'📈' if tipo_cortante == 'Positivo' else '📉'}")

# Calcular áreas por tramo para cortante
Nnodos_cor = LI_Geom.shape[0] + 1
Areas_Cortante = np.zeros((Nnodos_cor-1, 2))
for idx, tramo in enumerate(LI_Geom):
    tramo_indices = np.where((VInfo_Cortante[:, 0] >= tramo[0]) & (VInfo_Cortante[:, 0] <= tramo[1]))[0]
    tramo_info = VInfo_Cortante[tramo_indices, :]
    Areas_Cortante[idx, 0] = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] < 0])) * Step
    Areas_Cortante[idx, 1] = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] > 0])) * Step

st.markdown("#### 📋 Áreas por Tramo - Cortante")
areas_cor_df = pd.DataFrame(Areas_Cortante, columns=["Área Negativa", "Área Positiva"])
areas_cor_df.index = [f"Tramo {i+1}" for i in range(len(areas_cor_df))]
st.dataframe(areas_cor_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

# ============================================================
# BUSCAR ORDENADA DE CORTANTE EN CUALQUIER PUNTO X
# ============================================================
st.markdown("### 🔍 Consultar Ordenada de Cortante en Punto Específico")

st.info(f"ℹ️ Buscar ordenada de cortante: Ingresa cualquier posición X dentro del puente para obtener la ordenada de la línea de influencia de cortante en ese punto.")



col_buscar_cor1, col_buscar_cor2, col_buscar_cor3 = st.columns([1, 2, 1])

with col_buscar_cor2:
    x_consulta_cor = st.number_input(
        "📍 Posición X a consultar (m)",
        min_value=0.0,
        max_value=float(longitud_total),
        value=float(longitud_total / 2),
        step=0.1,
        key="x_consulta_cortante"
    )
    
    if st.button("🔎 Buscar Ordenada de Cortante", use_container_width=True, key="btn_buscar_cortante"):
        idx_consulta_cor = np.argmin(np.abs(VInfo_Cortante[:, 0] - x_consulta_cor))
        x_real_cor = VInfo_Cortante[idx_consulta_cor, 0]
        ordenada_consulta_cor = VInfo_Cortante[idx_consulta_cor, 1]
        
        st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**📍 Posición consultada:** X = {x_consulta_cor:.4f} m")
        st.markdown(f"**📏 Posición real más cercana:** X = {x_real_cor:.4f} m")
        st.markdown(f"**📊 Ordenada de LI Cortante:** {ordenada_consulta_cor:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.latex(f"L_i(x={x_real_cor:.4f}) = {ordenada_consulta_cor:.6f}")
        
        if ordenada_consulta_cor > 0:
            st.success(f"✅ La ordenada es **positiva** ({ordenada_consulta_cor:.6f})")
        elif ordenada_consulta_cor < 0:
            st.warning(f"⚠️ La ordenada es **negativa** ({ordenada_consulta_cor:.6f})")
        else:
            st.info(f"ℹ️ La ordenada es **cero** (punto de cambio de signo)")

st.markdown("---")

st.markdown("#### 🎚️ Selección de Tramos con Carga")
tramos_con_carga_cor = []
cols_cor = st.columns(len(Areas_Cortante))
for i, col in enumerate(cols_cor):
    with col:
        tramos_con_carga_cor.append(
            st.checkbox(f"Tramo {i+1}", value=True, key=f"tramo_cor_{i}")
        )

CortanteCargaCarril = np.zeros(len(tramos_con_carga_cor))

# Seleccionar área según tipo de cortante
idx_area = 1 if tipo_cortante == "Positivo" else 0  # 1 = positiva, 0 = negativa

for idx, tiene_carga in enumerate(tramos_con_carga_cor):
    if tiene_carga:
        # Usar área positiva o negativa según selección
        CortanteCargaCarril[idx] = Areas_Cortante[idx, idx_area] * factor_carga
    else:
        CortanteCargaCarril[idx] = 0

V_carril_total = np.sum(CortanteCargaCarril)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.metric(f"🎯 Cortante {tipo_cortante} Total", f"{V_carril_total:.4f} ton")

# Mostrar cálculos detallados
with st.expander("🧮 Ver Cálculos Detallados de Cortante por Carril", expanded=False):
    st.markdown(f"#### 📝 Procedimiento de Cálculo - Cortante {tipo_cortante} por Carril")
    
    area_tipo = "positivas" if tipo_cortante == "Positivo" else "negativas"
    st.markdown(f"**Paso 1: Áreas {area_tipo} bajo la línea de influencia de cortante**")
    st.latex(r"V = \sum (A_{" + area_tipo[:3] + r"} \times w)")
    
    st.markdown("**Paso 2: Cálculo por tramo**")
    
    area_sum = 0
    for i, (area_neg, area_pos) in enumerate(Areas_Cortante):
        if tramos_con_carga_cor[i]:
            area_usada = area_pos if tipo_cortante == "Positivo" else area_neg
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Tramo {i+1}:** ✅")
            with col2:
                st.latex(f"A_{{{area_tipo[:3]}}} = {area_usada:.4f}")
            
            area_sum += area_usada
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Tramo {i+1}:** ❌ (sin carga)")
    
    st.markdown(f"**Paso 3: Sumar áreas {area_tipo} de tramos con carga**")
    st.latex(f"A_{{{area_tipo[:3]},total}} = {area_sum:.4f}")
    
    st.markdown("**Paso 4: Aplicar intensidad de carga uniforme**")
    
    st.latex(f"w = {factor_carga:.3f} \\text{{ ton/m}}")
    
    st.latex(f"V = A_{{{area_tipo[:3]},total}} \\times w")
    
    st.latex(f"V = {area_sum:.4f} \\times {factor_carga:.3f}")
    
    st.markdown(f'<div class="result-box"><b>✅ Resultado Final - Cortante {tipo_cortante}:</b></div>', unsafe_allow_html=True)
    st.latex(f"V = {V_carril_total:.4f} \\text{{ ton}}")

# ============================================================
# 7️⃣ LÍNEA DE INFLUENCIA DE REACCIONES
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<h2 class="section-header">🏗️ Línea de Influencia de Reacciones</h2>', 
            unsafe_allow_html=True)

num_apoyos = LI_Geom.shape[0] + 1

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    Input2 = st.selectbox(
        "Selecciona el apoyo para análisis",
        options=list(range(1, num_apoyos + 1)),
        format_func=lambda x: f"Apoyo {x}"
    )

LIReac = LIREACCION(LI_Geom, int(Input2), Step)
fig_reac = LIReac.plot()

col_left4, col_graph4, col_right4 = st.columns([0.5, 3, 0.5])
with col_graph4:
    st.pyplot(fig_reac, use_container_width=False)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 8️⃣ RESUMEN Y EXPORTACIÓN
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📄 Resumen y Exportación</h2>', 
            unsafe_allow_html=True)

col_resumen1, col_resumen2 = st.columns(2)

with col_resumen1:
    st.markdown("### 📊 Resumen de Resultados")
    resumen_df = pd.DataFrame({
        "Parámetro": [
            "Momento HL-93",
            "Momento Carga de Carril",
            "Cortante HL-93",
            "Cortante Carga de Carril",
            "Punto de Análisis",
            "Longitud Total"
        ],
        "Valor": [
            f"{Mtotal:.4f} ton·m" if 'Mtotal' in locals() else "N/A",
            f"{M_carril_total:.4f} ton·m",
            f"{Vtotal:.4f} ton" if 'Vtotal' in locals() else "N/A",
            f"{V_carril_total:.4f} ton" if 'V_carril_total' in locals() else "N/A",
            f"{Panalisis:.2f} m",
            f"{longitud_total:.2f} m"
        ]
    })
    st.table(resumen_df)

with col_resumen2:
    st.markdown("### 📥 Exportar Resultados")
    
    if st.button("🎨 Generar Reporte PDF", use_container_width=True):
        with st.spinner("Generando PDF..."):
            figs = [
                fig_mom, 
                fig_hl93 if 'fig_hl93' in locals() else fig_mom, 
                fig_carril, 
                fig_cor,
                fig_hl93_cor if 'fig_hl93_cor' in locals() else fig_cor,
                fig_reac
            ]
            textos = [
                f"REPORTE DE LÍNEAS DE INFLUENCIA - {proyecto}",
                f"Analista: {analista}",
                f"Fecha: {fecha}",
                f"Punto de análisis: {Panalisis:.2f} m",
                f"Momento HL-93: {Mtotal:.4f} ton·m" if 'Mtotal' in locals() else "",
                f"Momento por carga de carril: {M_carril_total:.4f} ton·m",
                f"Cortante HL-93: {Vtotal:.4f} ton" if 'Vtotal' in locals() else "",
                f"Cortante por carga de carril: {V_carril_total:.4f} ton" if 'V_carril_total' in locals() else ""
            ]
            pdf_path = generar_pdf(figs, textos)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=f,
                    file_name=f"reporte_lineas_influencia_{proyecto.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        st.success("✅ PDF generado exitosamente")
    
    st.markdown("---")
    
    # Exportar datos a Excel
    st.markdown("### 📊 Exportar Datos a Excel")
    
    if st.button("📈 Generar Archivo Excel", use_container_width=True):
        with st.spinner("Generando archivo Excel..."):
            # Crear un objeto BytesIO para guardar el Excel en memoria
            from io import BytesIO
            output = BytesIO()
            
            # Crear el archivo Excel con múltiples hojas
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Hoja 1: Línea de Influencia de Momentos
                df_momentos = pd.DataFrame(VInfo, columns=['Distancia (m)', 'Ordenada LI Momento'])
                df_momentos.to_excel(writer, sheet_name='LI Momentos', index=False)
                
                # Hoja 2: Línea de Influencia de Cortantes
                df_cortantes = pd.DataFrame(VInfo_Cortante, columns=['Distancia (m)', 'Ordenada LI Cortante'])
                df_cortantes.to_excel(writer, sheet_name='LI Cortantes', index=False)
                
                # Hoja 3: Áreas por Tramo - Momentos
                areas_mom_export = pd.DataFrame(Areas, columns=["Área Negativa", "Área Positiva"])
                areas_mom_export.insert(0, 'Tramo', [f"Tramo {i+1}" for i in range(len(areas_mom_export))])
                areas_mom_export.to_excel(writer, sheet_name='Áreas Momentos', index=False)
                
                # Hoja 4: Áreas por Tramo - Cortantes
                areas_cor_export = pd.DataFrame(Areas_Cortante, columns=["Área Negativa", "Área Positiva"])
                areas_cor_export.insert(0, 'Tramo', [f"Tramo {i+1}" for i in range(len(areas_cor_export))])
                areas_cor_export.to_excel(writer, sheet_name='Áreas Cortantes', index=False)
                
                # Hoja 5: Resumen de Resultados
                resumen_export = pd.DataFrame({
                    "Parámetro": [
                        "Proyecto",
                        "Analista",
                        "Fecha",
                        "Punto de Análisis (m)",
                        "Longitud Total (m)",
                        "",
                        "MOMENTOS",
                        "Momento HL-93 (ton·m)",
                        "Momento Carga de Carril (ton·m)",
                        "",
                        "CORTANTES",
                        f"Cortante HL-93 {tipo_cortante_hl93} (ton)" if 'tipo_cortante_hl93' in locals() else "Cortante HL-93 (ton)",
                        f"Cortante Carga de Carril {tipo_cortante} (ton)" if 'tipo_cortante' in locals() else "Cortante Carga de Carril (ton)",
                    ],
                    "Valor": [
                        proyecto,
                        analista,
                        str(fecha),
                        f"{Panalisis:.2f}",
                        f"{longitud_total:.2f}",
                        "",
                        "",
                        f"{Mtotal:.4f}" if 'Mtotal' in locals() else "N/A",
                        f"{M_carril_total:.4f}",
                        "",
                        "",
                        f"{Vtotal:.4f}" if 'Vtotal' in locals() else "N/A",
                        f"{V_carril_total:.4f}" if 'V_carril_total' in locals() else "N/A",
                    ]
                })
                resumen_export.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja 6: Geometría del Puente
                geom_export = geom_df.copy()
                geom_export.to_excel(writer, sheet_name='Geometría', index=False)
            
            # Preparar el archivo para descarga
            output.seek(0)
            
            st.download_button(
                label="⬇️ Descargar Excel",
                data=output,
                file_name=f"datos_lineas_influencia_{proyecto.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.success("✅ Archivo Excel generado exitosamente")
        st.info("📋 El archivo contiene 6 hojas: LI Momentos, LI Cortantes, Áreas Momentos, Áreas Cortantes, Resumen y Geometría")
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🌉 <b>Análisis de Líneas de Influencia en Puentes</b></p>
    <p> Ingeniería Civil Estructural | Ñol ivan Jun de dios Rojas | © 2026 Todos los derechos reservados.</p>
</div>
""", unsafe_allow_html=True)