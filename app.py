# -*- coding: utf-8 -*-
"""
FinanziaStock Pro — Sistema de Predicción de Demanda con IA
Dashboard interactivo para distribuidores mayoristas y PYMES
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_manager import (save_empresa_info, load_empresa_info, list_empresas,
                                 upload_data, load_consolidated_data, get_empresa_dir)
from utils.model_engine import train_model, predict_future, test_model_health, retrain_with_new_data

st.set_page_config(
    page_title="FinanziaStock Pro | Predicción de Demanda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DISEÑO PROFESIONAL =====================
st.markdown("""
<style>
    /* Ocultar header y footer default de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Layout */
    .block-container {padding: 1rem 2rem 2rem 2rem; max-width: 1400px;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003366 0%, #001a33 100%);
    }
    [data-testid="stSidebar"] * {color: white !important;}
    [data-testid="stSidebar"] .stSelectbox label {color: rgba(255,255,255,0.7) !important; font-size: 0.85rem;}
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,0.1);}
    [data-testid="stSidebar"] .stRadio label {font-size: 0.9rem;}
    
    /* Métricas */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] {font-size: 0.8rem; color: #6b7280;}
    [data-testid="stMetricValue"] {font-size: 1.5rem; color: #003366; font-weight: 700;}
    
    /* Headers */
    h1 {color: #003366 !important; font-weight: 800; letter-spacing: -0.5px;}
    h2, h3 {color: #003366 !important; font-weight: 700;}
    
    /* Botones */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #003366, #0078B4);
        border: none;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #001a33, #005f8f);
        box-shadow: 0 4px 12px rgba(0,51,102,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Tablas */
    .stDataFrame {border-radius: 12px; overflow: hidden;}
    
    /* Cards de info */
    .info-card {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .info-card h4 {color: #003366; margin-bottom: 8px;}
    
    /* Alertas */
    .alert-good {background: #d4edda; border-left: 4px solid #2a9d8f; padding: 16px; border-radius: 0 10px 10px 0; margin: 10px 0;}
    .alert-warn {background: #fff3cd; border-left: 4px solid #e9c46a; padding: 16px; border-radius: 0 10px 10px 0; margin: 10px 0;}
    .alert-bad {background: #f8d7da; border-left: 4px solid #e63946; padding: 16px; border-radius: 0 10px 10px 0; margin: 10px 0;}
    .alert-info {background: #e8f4f8; border-left: 4px solid #0078B4; padding: 16px; border-radius: 0 10px 10px 0; margin: 10px 0;}
    
    /* Logo en sidebar */
    .logo-container {text-align: center; padding: 20px 0 10px 0;}
    .logo-container h2 {font-size: 1.4rem; margin: 0; letter-spacing: -0.5px;}
    .logo-container p {font-size: 0.75rem; opacity: 0.6; margin: 4px 0 0 0;}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <h2>📊 FinanziaStock Pro</h2>
        <p>Predicción de Demanda con IA</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Empresa selector
    empresas = list_empresas()
    if empresas:
        nit_options = {f"{e['nombre']}": e['nit'] for e in empresas}
        selected = st.selectbox("🏢 Empresa activa", list(nit_options.keys()))
        nit = nit_options[selected]
        info = load_empresa_info(nit)
        if info:
            st.caption(f"NIT: {info.get('nit','N/A')} | {info.get('total_registros',0):,} registros")
    else:
        nit = None
        st.info("Registre una empresa para comenzar")
    
    st.markdown("---")
    st.markdown("##### Navegación")
    st.page_link("app.py", label="🏠 Inicio", icon="🏠")
    st.page_link("pages/1_📋_Empresas.py", label="📋 Empresas")
    st.page_link("pages/2_📤_Subir_Datos.py", label="📤 Subir Datos")
    st.page_link("pages/3_🔮_Prediccion.py", label="🔮 Predicción")
    st.page_link("pages/4_🧪_Testing.py", label="🧪 Testing Modelo")
    st.page_link("pages/5_🔄_Reentrenamiento.py", label="🔄 Reentrenamiento")
    st.page_link("pages/6_📊_Backtesting.py", label="📊 Backtesting")
    st.page_link("pages/7_📄_Informe.py", label="📄 Informes")
    st.page_link("pages/8_📚_Historial.py", label="📚 Historial")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;opacity:0.5;font-size:0.7rem;padding:10px 0'>
    FinanziaStock Pro v2.0<br>
    Machine Learning + XGBoost<br>
    © 2026 Dr. Julián Cucalón
    </div>
    """, unsafe_allow_html=True)

# ===================== PÁGINA PRINCIPAL =====================
st.markdown("# 📊 FinanziaStock Pro")
st.markdown("##### Sistema de Predicción de Demanda con Inteligencia Artificial")
st.markdown("---")

if empresas and nit:
    df = load_consolidated_data(nit)
    
    if df is not None and len(df) > 0:
        # === MÉTRICAS PRINCIPALES ===
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📦 Productos", df['producto'].nunique())
        with col2:
            st.metric("📋 Registros", f"{len(df):,}")
        with col3:
            if 'valor_total' in df.columns:
                ventas = df['valor_total'].sum()
                st.metric("💰 Ventas totales", f"${ventas/1e6:.1f}M")
            else:
                st.metric("📊 Cantidad total", f"{df['cantidad'].sum():,.0f}")
        with col4:
            if 'cliente' in df.columns:
                st.metric("🏢 Clientes", df['cliente'].nunique())
            else:
                st.metric("📅 Días de datos", df['fecha'].nunique())
        with col5:
            periodo = f"{df['fecha'].min().strftime('%Y-%m')}"
            st.metric("📅 Desde", periodo)
        
        st.markdown("")
        
        # === GRÁFICOS ===
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Ventas diarias
            ventas_dia = df.groupby('fecha')['cantidad'].sum().reset_index()
            fig = px.area(ventas_dia, x='fecha', y='cantidad',
                        title='📈 Ventas diarias totales',
                        labels={'cantidad': 'Cantidad', 'fecha': 'Fecha'})
            fig.update_traces(fill='tozeroy', line_color='#003366', fillcolor='rgba(0,51,102,0.1)')
            fig.update_layout(title_font_color='#003366', title_font_size=16, 
                            plot_bgcolor='white', paper_bgcolor='white',
                            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            # Top productos
            top = df.groupby('producto')['cantidad'].sum().sort_values(ascending=True).tail(8)
            fig = px.bar(x=top.values, y=top.index, orientation='h',
                        title='📦 Top productos por volumen',
                        labels={'x': 'Cantidad total', 'y': 'Producto'})
            fig.update_traces(marker_color='#003366')
            fig.update_layout(title_font_color='#003366', title_font_size=16,
                            plot_bgcolor='white', paper_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
        
        # === ACCIONES RÁPIDAS ===
        st.markdown("### ⚡ Acciones rápidas")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>🔮 Predecir demanda</h4>
                <p style="font-size:0.85rem;color:#666">Genere predicciones para los próximos 7-90 días por producto</p>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/3_🔮_Prediccion.py", label="Ir a Predicción →")
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>📊 Backtesting</h4>
                <p style="font-size:0.85rem;color:#666">Valide la precisión del modelo con datos históricos (0-100%)</p>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/6_📊_Backtesting.py", label="Ir a Backtesting →")
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4>📤 Subir datos nuevos</h4>
                <p style="font-size:0.85rem;color:#666">Cargue el archivo del mes para mantener el modelo actualizado</p>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/2_📤_Subir_Datos.py", label="Ir a Subir Datos →")
        with col4:
            st.markdown("""
            <div class="info-card">
                <h4>📄 Generar informe</h4>
                <p style="font-size:0.85rem;color:#666">Informe ejecutivo con predicciones y recomendaciones de compra</p>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/7_📄_Informe.py", label="Ir a Informes →")
        
        # === INFO DEL MODELO ===
        empresa_dir = get_empresa_dir(nit)
        meta_path = os.path.join(empresa_dir, 'modelos')
        if os.path.exists(meta_path):
            meta_files = [f for f in os.listdir(meta_path) if f.startswith('meta_')]
            if meta_files:
                st.markdown("### 🤖 Modelos entrenados")
                cols = st.columns(min(len(meta_files), 4))
                for i, mf in enumerate(meta_files[:4]):
                    with open(os.path.join(meta_path, mf), 'r') as f:
                        meta = json.load(f)
                    with cols[i]:
                        producto = meta.get('producto', mf.replace('meta_','').replace('.json',''))
                        metricas = meta.get('metricas', {})
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>{producto}</h4>
                            <p style="font-size:0.8rem;color:#666">
                            MAE: <strong>{metricas.get('MAE','N/A')}</strong> | 
                            MAPE: <strong>{metricas.get('MAPE','N/A')}%</strong><br>
                            Entrenado: {meta.get('fecha_entrenamiento','N/A')[:10]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        # Sin datos
        st.markdown("""
        <div class="alert-info">
            <h4>👋 Bienvenido a FinanziaStock Pro</h4>
            <p>Para comenzar, siga estos pasos:</p>
            <ol>
                <li><strong>Registre su empresa</strong> en la sección 📋 Empresas (NIT, razón social)</li>
                <li><strong>Suba sus datos de ventas</strong> en la sección 📤 Subir Datos (Excel o CSV)</li>
                <li><strong>Entrene el modelo</strong> en la sección 🔮 Predicción</li>
                <li><strong>Genere predicciones</strong> y tome decisiones basadas en datos</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📎 ¿Qué datos necesita?")
        st.markdown("""
        Un archivo **Excel (.xlsx) o CSV** con mínimo estas columnas:
        
        | fecha | producto | cantidad |
        |-------|----------|----------|
        | 2024-01-05 | Arroz 50kg | 24 |
        | 2024-01-05 | Aceite x12 | 12 |
        
        **Columnas opcionales** que mejoran la precisión: `valor_total`, `cliente`, `en_promocion`, `categoria`, `precio_unitario`, `ciudad`, `vendedor`
        """)
else:
    # Sin empresas
    st.markdown("""
    <div style="text-align:center; padding:60px 0">
        <h1 style="font-size:4rem; margin-bottom:0">📊</h1>
        <h2>Bienvenido a FinanziaStock Pro</h2>
        <p style="color:#666; font-size:1.1rem; max-width:600px; margin:0 auto 30px auto">
            Sistema de Predicción de Demanda con Inteligencia Artificial para 
            distribuidores mayoristas y PYMES colombianas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card" style="text-align:center">
            <div style="font-size:2.5rem">🤖</div>
            <h4>Predicción con IA</h4>
            <p style="font-size:0.85rem;color:#666">XGBoost predice la demanda de cada producto para los próximos 7-90 días</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align:center">
            <div style="font-size:2.5rem">📊</div>
            <h4>Backtesting</h4>
            <p style="font-size:0.85rem;color:#666">Valide la precisión del modelo comparando predicciones vs datos reales</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align:center">
            <div style="font-size:2.5rem">📋</div>
            <h4>Informes automáticos</h4>
            <p style="font-size:0.85rem;color:#666">Genere informes ejecutivos con recomendaciones de compra descargables</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.page_link("pages/1_📋_Empresas.py", label="🚀 Comenzar — Registrar mi empresa", icon="🚀")
