# -*- coding: utf-8 -*-
"""Página: Reentrenamiento con datos nuevos del mes."""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, load_consolidated_data, upload_data
from utils.model_engine import retrain_with_new_data, test_model_health

st.set_page_config(page_title="Reentrenamiento | Demand Forecast", page_icon="🔄", layout="wide")
st.title("🔄 Reentrenamiento del Modelo")

st.markdown("""
### Proceso de actualización mensual
1. **Suba los datos nuevos del mes** (Excel/CSV)
2. **Los datos se consolidan** automáticamente con el historial existente
3. **El modelo se reentrena** con todos los datos actualizados
4. **Se evalúa** el nuevo modelo vs el anterior
""")

empresas = list_empresas()
if not empresas:
    st.warning("⚠️ Registre una empresa primero"); st.stop()

nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Empresa:", list(nit_options.keys()))
nit = nit_options[selected]

st.markdown("---")

# Paso 1: Subir datos nuevos
st.subheader("📤 Paso 1: Subir datos nuevos del mes")
uploaded = st.file_uploader("Archivo de ventas del mes", type=['csv','xlsx'])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded)
        else:
            df_new = pd.read_excel(uploaded)
        
        st.dataframe(df_new.head(10), use_container_width=True)
        st.info(f"📊 {len(df_new):,} registros nuevos detectados")
        
        if st.button("✅ Consolidar datos nuevos", type="primary"):
            result = upload_data(nit, df_new, uploaded.name)
            if result['ok']:
                st.success(f"✅ Datos consolidados: {result['total_consolidado']:,} registros totales")
            else:
                st.error(result['error'])
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Paso 2: Reentrenar
st.subheader("🔄 Paso 2: Reentrenar modelos")
df = load_consolidated_data(nit)
if df is not None:
    productos = sorted(df['producto'].unique())
    
    retrain_all = st.checkbox("Reentrenar TODOS los productos", value=False)
    if not retrain_all:
        producto_sel = st.selectbox("Producto a reentrenar:", productos)
        prods = [producto_sel]
    else:
        prods = productos
    
    if st.button("🚀 Reentrenar", type="primary", use_container_width=True):
        progress = st.progress(0)
        results = []
        
        for i, prod in enumerate(prods):
            with st.spinner(f"Reentrenando {prod}..."):
                result = retrain_with_new_data(nit, df, prod)
                if result['ok']:
                    results.append({
                        'Producto': prod, 'Estado': '✅ OK',
                        'MAE': result['metricas']['MAE'],
                        'MAPE': f"{result['metricas']['MAPE']}%",
                        'R²': result['metricas']['R2'],
                    })
                else:
                    results.append({'Producto': prod, 'Estado': f"❌ {result.get('error','')}", 
                                   'MAE':'-','MAPE':'-','R²':'-'})
            progress.progress((i+1)/len(prods))
        
        df_results = pd.DataFrame(results)
        st.subheader("Resultados del reentrenamiento")
        st.dataframe(df_results, use_container_width=True)
        
        n_ok = sum(1 for r in results if '✅' in r['Estado'])
        st.success(f"✅ {n_ok}/{len(prods)} modelos reentrenados exitosamente")
else:
    st.warning("No hay datos consolidados. Suba datos primero.")
