# -*- coding: utf-8 -*-
"""Página: Testing y monitoreo de salud del modelo."""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, load_consolidated_data
from utils.model_engine import test_model_health

st.set_page_config(page_title="Testing | Demand Forecast", page_icon="🧪", layout="wide")
st.title("🧪 Testing y Monitoreo del Modelo")

st.markdown("""
Este módulo detecta **degradación del modelo** comparando su rendimiento actual con el rendimiento 
al momento del entrenamiento. Si la degradación supera el 20%, se recomienda reentrenar.
""")

empresas = list_empresas()
if not empresas:
    st.warning("⚠️ Registre una empresa primero"); st.stop()

nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Empresa:", list(nit_options.keys()))
nit = nit_options[selected]

df = load_consolidated_data(nit)
if df is None:
    st.warning("⚠️ No hay datos"); st.stop()

st.markdown("---")

productos = sorted(df['producto'].unique())
test_all = st.checkbox("Testear TODOS los productos", value=True)

if test_all:
    prods_to_test = productos
else:
    prods_to_test = [st.selectbox("Producto:", productos)]

if st.button("🧪 Ejecutar Testing", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0)
    
    for i, prod in enumerate(prods_to_test):
        result = test_model_health(nit, df, prod)
        if result['ok']:
            results.append({
                'Producto': prod,
                'Salud': result['salud'],
                'MAE Original': result['mae_original'],
                'MAE Actual': result['mae_actual'],
                'MAPE Actual': f"{result['mape_actual']}%",
                'Degradación': f"{result['degradacion_pct']}%",
                'Días sin reentrenar': result['dias_desde_entrenamiento'],
                'Reentrenar?': '🔴 SÍ' if result['necesita_reentrenamiento'] else '🟢 NO',
            })
        else:
            results.append({
                'Producto': prod, 'Salud': '⚪ SIN MODELO', 
                'MAE Original': '-', 'MAE Actual': '-', 'MAPE Actual': '-',
                'Degradación': '-', 'Días sin reentrenar': '-', 'Reentrenar?': '⚪ Entrenar primero'
            })
        progress.progress((i+1)/len(prods_to_test))
    
    df_results = pd.DataFrame(results)
    
    # Resumen
    n_good = sum(1 for r in results if r['Salud'] == 'BUENO')
    n_degraded = sum(1 for r in results if r['Salud'] == 'DEGRADADO')
    n_critical = sum(1 for r in results if r['Salud'] == 'CRITICO')
    n_none = sum(1 for r in results if 'SIN MODELO' in str(r['Salud']))
    
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("🟢 Saludables", n_good)
    col2.metric("🟡 Degradados", n_degraded)
    col3.metric("🔴 Críticos", n_critical)
    col4.metric("⚪ Sin modelo", n_none)
    
    st.markdown("### Resultados detallados")
    st.dataframe(df_results, use_container_width=True)
    
    if n_degraded + n_critical > 0:
        st.warning(f"⚠️ {n_degraded + n_critical} producto(s) necesitan reentrenamiento. Vaya a la sección 🔄 Reentrenamiento.")
