# -*- coding: utf-8 -*-
"""
🔮 Demand Forecast Pro - Sistema de Predicción de Demanda
Dashboard interactivo para distribuidores mayoristas y PYMES
Ejecutar: streamlit run app.py
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

st.set_page_config(page_title="Demand Forecast Pro", page_icon="🔮", layout="wide")

# === CSS ===
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    .stMetric {border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px;}
    h1 {color: #003366 !important;}
    h2, h3 {color: #003366 !important;}
    .health-good {background:#d4edda; padding:15px; border-radius:8px; border-left:5px solid #28a745;}
    .health-warn {background:#fff3cd; padding:15px; border-radius:8px; border-left:5px solid #ffc107;}
    .health-bad {background:#f8d7da; padding:15px; border-radius:8px; border-left:5px solid #dc3545;}
</style>
""", unsafe_allow_html=True)

# === SIDEBAR: Empresa ===
st.sidebar.title("🔮 Demand Forecast Pro")
st.sidebar.markdown("---")

# Empresa selection/creation
empresas = list_empresas()
empresa_names = {e['nit']: f"{e['nombre']} ({e['nit']})" for e in empresas}

st.sidebar.subheader("🏢 Empresa")
modo = st.sidebar.radio("", ["Seleccionar existente", "Registrar nueva"], label_visibility="collapsed")

if modo == "Registrar nueva":
    with st.sidebar.form("nueva_empresa"):
        nit = st.text_input("NIT", placeholder="900.123.456-7")
        nombre = st.text_input("Nombre", placeholder="Distribuidora XYZ S.A.S")
        sector = st.selectbox("Sector", ["Alimentos","Bebidas","Aseo","Construcción","Tecnología","Farmacéutico","Otro"])
        contacto = st.text_input("Contacto", placeholder="email@empresa.com")
        if st.form_submit_button("✅ Registrar", type="primary"):
            if nit and nombre:
                save_empresa_info(nit, nombre, sector, contacto)
                st.sidebar.success(f"Empresa {nombre} registrada")
                st.rerun()
            else:
                st.sidebar.error("NIT y Nombre son obligatorios")
    st.stop()
else:
    if not empresas:
        st.sidebar.warning("No hay empresas. Registre una primero.")
        st.stop()
    nit_sel = st.sidebar.selectbox("Empresa:", list(empresa_names.keys()),
                                    format_func=lambda x: empresa_names[x])
    empresa_info = load_empresa_info(nit_sel)

st.sidebar.markdown("---")

# Navigation
st.sidebar.subheader("📋 Módulos")
pagina = st.sidebar.radio("", [
    "📊 Dashboard",
    "📤 Subir Datos",
    "🤖 Entrenar Modelo",
    "🔮 Predicción",
    "🧪 Testing del Modelo",
    "🔄 Reentrenamiento"
], label_visibility="collapsed")

# === LOAD DATA ===
df = load_consolidated_data(nit_sel)

# ================================================================
# PÁGINA: DASHBOARD
# ================================================================
if pagina == "📊 Dashboard":
    st.title(f"📊 Dashboard - {empresa_info.get('nombre','')}")
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No hay datos cargados. Vaya a **📤 Subir Datos** para comenzar.")
        st.stop()
    
    # Métricas
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📋 Registros", f"{len(df):,}")
    c2.metric("📦 Productos", f"{df['producto'].nunique()}")
    c3.metric("📅 Periodo", f"{(df['fecha'].max()-df['fecha'].min()).days} días")
    if 'cliente' in df.columns:
        c4.metric("🏢 Clientes", f"{df['cliente'].nunique()}")
    if 'valor_total' in df.columns:
        c5.metric("💰 Ventas", f"${df['valor_total'].sum()/1e6:.0f}M")
    
    st.markdown("---")
    col1,col2 = st.columns(2)
    
    with col1:
        ventas_dia = df.groupby('fecha')['cantidad'].sum().reset_index()
        fig = px.line(ventas_dia, x='fecha', y='cantidad', title='Cantidad Total por Día')
        fig.update_traces(line_color='#003366')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top = df.groupby('producto')['cantidad'].sum().sort_values(ascending=True).tail(10)
        fig = px.bar(x=top.values, y=top.index, orientation='h', title='Top 10 Productos')
        fig.update_traces(marker_color='#003366')
        st.plotly_chart(fig, use_container_width=True)
    
    col3,col4 = st.columns(2)
    with col3:
        dias={0:'Lun',1:'Mar',2:'Mié',3:'Jue',4:'Vie',5:'Sáb',6:'Dom'}
        pat = df.copy(); pat['dia'] = pat['fecha'].dt.dayofweek.map(dias)
        pat_g = pat.groupby('dia')['cantidad'].mean().reset_index()
        fig = px.bar(pat_g, x='dia', y='cantidad', title='Patrón Semanal')
        fig.update_traces(marker_color='#2a9d8f')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        pat_m = df.copy(); pat_m['mes'] = pat_m['fecha'].dt.month
        pat_mg = pat_m.groupby('mes')['cantidad'].sum().reset_index()
        fig = px.bar(pat_mg, x='mes', y='cantidad', title='Ventas por Mes')
        fig.update_traces(marker_color='#e63946')
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PÁGINA: SUBIR DATOS
# ================================================================
elif pagina == "📤 Subir Datos":
    st.title("📤 Subir Datos de Ventas")
    st.markdown(f"**Empresa:** {empresa_info.get('nombre','')} | **NIT:** {nit_sel}")
    
    st.info("""
    **Formato esperado del archivo (CSV o Excel):**
    
    | fecha | producto | cantidad | valor_total | cliente |
    |-------|----------|----------|-------------|---------|
    | 2024-01-05 | Arroz 50kg | 24 | 1440000 | CLI-001 |
    
    - **fecha** y **producto** y **cantidad** son obligatorias
    - **valor_total** y **cliente** son opcionales pero recomendadas
    """)
    
    archivo = st.file_uploader("📎 Seleccione archivo CSV o Excel", type=['csv','xlsx','xls'])
    
    if archivo:
        try:
            if archivo.name.endswith('.csv'):
                df_new = pd.read_csv(archivo)
            else:
                df_new = pd.read_excel(archivo)
            
            st.subheader("Vista previa")
            st.dataframe(df_new.head(10), use_container_width=True)
            st.write(f"**Filas:** {len(df_new):,} | **Columnas:** {', '.join(df_new.columns)}")
            
            if st.button("✅ Confirmar y Subir", type="primary", use_container_width=True):
                with st.spinner("Procesando datos..."):
                    result = upload_data(nit_sel, df_new, archivo.name)
                
                if result['ok']:
                    st.success(f"""
                    ✅ **Datos subidos exitosamente**
                    - Registros nuevos: {result['registros_nuevos']:,}
                    - Total consolidado: {result['total_consolidado']:,}
                    - Productos: {result['productos']}
                    - Periodo: {result['periodo']}
                    """)
                    st.balloons()
                else:
                    st.error(f"❌ Error: {result['error']}")
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

# ================================================================
# PÁGINA: ENTRENAR MODELO
# ================================================================
elif pagina == "🤖 Entrenar Modelo":
    st.title("🤖 Entrenar Modelo de Predicción")
    
    if df is None:
        st.warning("Primero suba datos en 📤 Subir Datos"); st.stop()
    
    productos = sorted(df['producto'].unique())
    producto_sel = st.selectbox("Seleccione producto:", productos)
    
    n_dias = len(df[df['producto']==producto_sel]['fecha'].unique())
    st.write(f"Datos disponibles: **{n_dias} días** de ventas para este producto")
    
    if n_dias < 60:
        st.warning(f"Se necesitan mínimo 60 días de datos. Tiene {n_dias}.")
    
    test_pct = st.slider("% datos para testing:", 10, 30, 20)
    
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo XGBoost... (puede tomar 15-30 segundos)"):
            result = train_model(nit_sel, df, producto_sel, test_ratio=test_pct/100)
        
        if result['ok']:
            st.success("✅ Modelo entrenado exitosamente")
            
            # Métricas
            m = result['metricas']
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("MAE", f"{m['MAE']:.1f}")
            c2.metric("RMSE", f"{m['RMSE']:.1f}")
            c3.metric("R²", f"{m['R2']:.3f}")
            c4.metric("MAPE", f"{m['MAPE']:.1f}%")
            
            # Gráfico pred vs real
            pvr = result['pred_vs_real']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pvr['fechas'],y=pvr['real'],name='Real',line=dict(color='#003366')))
            fig.add_trace(go.Scatter(x=pvr['fechas'],y=pvr['prediccion'],name='Predicción',line=dict(color='#e63946',dash='dash')))
            fig.update_layout(title=f'Predicción vs Real - {producto_sel}',xaxis_title='Fecha',yaxis_title='Cantidad')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            fi = result['feature_importance']
            fi_df = pd.DataFrame({'Feature':fi.keys(),'Importancia':fi.values()}).sort_values('Importancia',ascending=True)
            fig = px.bar(fi_df, x='Importancia',y='Feature',orientation='h',title='Importancia de Variables')
            fig.update_traces(marker_color='#003366')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ {result['error']}")

# ================================================================
# PÁGINA: PREDICCIÓN
# ================================================================
elif pagina == "🔮 Predicción":
    st.title("🔮 Predicción de Demanda Futura")
    
    if df is None:
        st.warning("Primero suba datos"); st.stop()
    
    productos = sorted(df['producto'].unique())
    col1,col2 = st.columns(2)
    with col1:
        producto_sel = st.selectbox("Producto:", productos)
    with col2:
        dias = st.slider("Días a predecir:", 7, 90, 30)
    
    if st.button("🔮 Generar Predicción", type="primary", use_container_width=True):
        with st.spinner("Generando predicción..."):
            result = predict_future(nit_sel, df, producto_sel, dias)
        
        if result['ok']:
            preds = pd.DataFrame(result['predicciones'])
            total = result['total']
            
            c1,c2,c3 = st.columns(3)
            c1.metric("📦 Total a comprar", f"{total:.0f} unidades")
            c2.metric("📊 Promedio diario", f"{total/len(preds):.0f} uds/día")
            c3.metric("📅 Días predichos", f"{len(preds)}")
            
            # Gráfico
            hist = df[df['producto']==producto_sel].groupby('fecha')['cantidad'].sum().reset_index().tail(60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['fecha'],y=hist['cantidad'],name='Histórico',line=dict(color='#003366',width=2)))
            fig.add_trace(go.Scatter(x=pd.to_datetime(preds['fecha']),y=preds['prediccion'],
                                     name='Predicción',line=dict(color='#e63946',width=2,dash='dash')))
            fig.update_layout(title=f'Predicción {dias} días - {producto_sel}',hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detalle")
            st.dataframe(preds, use_container_width=True, height=300)
            
            st.success(f"**Recomendación:** Comprar ~**{total:.0f} unidades** de {producto_sel} para los próximos {dias} días")
        else:
            st.error(f"❌ {result['error']}")

# ================================================================
# PÁGINA: TESTING DEL MODELO
# ================================================================
elif pagina == "🧪 Testing del Modelo":
    st.title("🧪 Testing y Salud del Modelo")
    st.markdown("Verifica si el modelo sigue siendo preciso o necesita reentrenamiento.")
    
    if df is None:
        st.warning("Primero suba datos"); st.stop()
    
    productos = sorted(df['producto'].unique())
    producto_sel = st.selectbox("Producto a evaluar:", productos)
    
    if st.button("🧪 Ejecutar Test de Salud", type="primary", use_container_width=True):
        with st.spinner("Evaluando modelo..."):
            result = test_model_health(nit_sel, df, producto_sel)
        
        if result['ok']:
            salud = result['salud']
            
            if salud == 'BUENO':
                st.markdown(f"""<div class="health-good">
                    <h3>✅ Estado: BUENO</h3>
                    <p>El modelo funciona correctamente. No requiere reentrenamiento.</p>
                </div>""", unsafe_allow_html=True)
            elif salud == 'DEGRADADO':
                st.markdown(f"""<div class="health-warn">
                    <h3>⚠️ Estado: DEGRADADO</h3>
                    <p>El modelo ha perdido precisión. Se recomienda reentrenar con datos recientes.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="health-bad">
                    <h3>🚨 Estado: CRÍTICO</h3>
                    <p>El modelo está significativamente degradado. Reentrenamiento URGENTE.</p>
                </div>""", unsafe_allow_html=True)
            
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("MAE Original", f"{result['mae_original']:.1f}")
            c2.metric("MAE Actual", f"{result['mae_actual']:.1f}")
            c3.metric("Degradación", f"{result['degradacion_pct']:+.1f}%")
            c4.metric("Días desde entreno", f"{result['dias_desde_entrenamiento']}")
            
            if result['necesita_reentrenamiento']:
                st.warning("💡 **Recomendación:** Vaya a 🔄 Reentrenamiento para actualizar el modelo")
        else:
            st.error(f"❌ {result['error']}")

# ================================================================
# PÁGINA: REENTRENAMIENTO
# ================================================================
elif pagina == "🔄 Reentrenamiento":
    st.title("🔄 Reentrenamiento del Modelo")
    st.markdown("""
    Suba los datos nuevos del mes y reentrene el modelo para mantener la precisión.
    
    **Proceso:**
    1. Suba el archivo con datos del nuevo periodo
    2. Los datos se agregan al historial existente
    3. El modelo se reentrena con TODOS los datos acumulados
    """)
    
    if df is None:
        st.warning("Primero suba datos"); st.stop()
    
    st.subheader("1️⃣ Subir datos nuevos")
    archivo_nuevo = st.file_uploader("📎 Archivo con datos del nuevo periodo", type=['csv','xlsx'])
    
    if archivo_nuevo:
        if archivo_nuevo.name.endswith('.csv'):
            df_new = pd.read_csv(archivo_nuevo)
        else:
            df_new = pd.read_excel(archivo_nuevo)
        st.dataframe(df_new.head(5), use_container_width=True)
        
        if st.button("📤 Agregar datos al historial"):
            result = upload_data(nit_sel, df_new, archivo_nuevo.name)
            if result['ok']:
                st.success(f"✅ {result['registros_nuevos']} registros agregados. Total: {result['total_consolidado']}")
                df = load_consolidated_data(nit_sel)  # recargar
    
    st.markdown("---")
    st.subheader("2️⃣ Reentrenar modelo")
    
    if df is not None:
        productos = sorted(df['producto'].unique())
        producto_sel = st.selectbox("Producto a reentrenar:", productos)
        
        if st.button("🔄 Reentrenar con todos los datos", type="primary", use_container_width=True):
            with st.spinner("Reentrenando modelo..."):
                result = retrain_with_new_data(nit_sel, df, producto_sel)
            
            if result['ok']:
                st.success("✅ Modelo reentrenado exitosamente")
                m = result['metricas']
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("MAE", f"{m['MAE']:.1f}")
                c2.metric("RMSE", f"{m['RMSE']:.1f}")
                c3.metric("R²", f"{m['R2']:.3f}")
                c4.metric("MAPE", f"{m['MAPE']:.1f}%")
                st.balloons()
            else:
                st.error(f"❌ {result['error']}")

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.caption("🔮 Demand Forecast Pro v1.0")
st.sidebar.caption("ML: XGBoost | UI: Streamlit")
