# -*- coding: utf-8 -*-
"""Página: Subir datos Excel/CSV."""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, upload_data, load_consolidated_data

st.set_page_config(page_title="Subir Datos | Demand Forecast", page_icon="📤", layout="wide")
st.title("📤 Subir Datos de Ventas")

empresas = list_empresas()
if not empresas:
    st.warning("⚠️ Primero registre una empresa en la sección 📋 Empresas")
    st.stop()

# Seleccionar empresa
nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Seleccione empresa:", list(nit_options.keys()))
nit = nit_options[selected]

st.markdown("---")

# Upload
st.subheader("📎 Cargar archivo de ventas")
st.markdown("""
**Formatos aceptados:** CSV, XLSX (Excel)  
**Columnas obligatorias:** `fecha`, `producto`, `cantidad`  
**Columnas opcionales:** `valor_total`, `cliente`, `en_promocion`
""")

uploaded = st.file_uploader("Seleccione archivo", type=['csv','xlsx'], key='upload_ventas')

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(20), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", f"{len(df):,}")
        col2.metric("Columnas", len(df.columns))
        col3.metric("Columnas detectadas", ", ".join(df.columns[:5]))
        
        if st.button("✅ Confirmar y subir datos", type="primary", use_container_width=True):
            with st.spinner("Procesando datos..."):
                result = upload_data(nit, df, uploaded.name)
            
            if result['ok']:
                st.success(f"""
                ✅ **Datos cargados exitosamente**
                - Registros nuevos: **{result['registros_nuevos']:,}**
                - Total consolidado: **{result['total_consolidado']:,}**
                - Productos: **{result['productos']}**
                - Periodo: **{result['periodo']}**
                """)
                st.balloons()
            else:
                st.error(f"❌ Error: {result['error']}")
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")

# Mostrar datos existentes
st.markdown("---")
st.subheader("📊 Datos existentes de esta empresa")
df_existing = load_consolidated_data(nit)
if df_existing is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total registros", f"{len(df_existing):,}")
    col2.metric("Productos", df_existing['producto'].nunique())
    col3.metric("Desde", df_existing['fecha'].min().strftime('%Y-%m-%d') if hasattr(df_existing['fecha'].iloc[0],'strftime') else str(df_existing['fecha'].min())[:10])
    col4.metric("Hasta", df_existing['fecha'].max().strftime('%Y-%m-%d') if hasattr(df_existing['fecha'].iloc[0],'strftime') else str(df_existing['fecha'].max())[:10])
    
    with st.expander("Ver últimos registros"):
        st.dataframe(df_existing.tail(50), use_container_width=True)
else:
    st.info("No hay datos cargados para esta empresa. Suba un archivo arriba.")
