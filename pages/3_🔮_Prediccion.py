# -*- coding: utf-8 -*-
"""Página: Predicción de Demanda."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, load_consolidated_data
from utils.model_engine import train_model, predict_future

st.set_page_config(page_title="Predicción | Demand Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Predicción de Demanda")

empresas = list_empresas()
if not empresas:
    st.warning("⚠️ Registre una empresa y suba datos primero"); st.stop()

nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Empresa:", list(nit_options.keys()))
nit = nit_options[selected]

df = load_consolidated_data(nit)
if df is None:
    st.warning("⚠️ No hay datos para esta empresa. Suba datos primero."); st.stop()

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    producto = st.selectbox("Producto:", sorted(df['producto'].unique()))
with col2:
    dias = st.slider("Días a predecir:", 7, 90, 30)
with col3:
    st.metric("Registros disponibles", f"{len(df[df['producto']==producto]):,}")

tab1, tab2 = st.tabs(["🏋️ Entrenar Modelo", "🔮 Generar Predicción"])

with tab1:
    st.subheader("Entrenar modelo para este producto")
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner(f"Entrenando modelo para {producto}..."):
            result = train_model(nit, df, producto)
        
        if result['ok']:
            st.success("✅ Modelo entrenado exitosamente")
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("MAE", result['metricas']['MAE'])
            col2.metric("MAPE", f"{result['metricas']['MAPE']}%")
            col3.metric("R²", result['metricas']['R2'])
            col4.metric("RMSE", result['metricas']['RMSE'])
            
            # Gráfico pred vs real
            pvr = result['pred_vs_real']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pvr['fechas'], y=pvr['real'], name='Real', line=dict(color='#003366',width=2)))
            fig.add_trace(go.Scatter(x=pvr['fechas'], y=pvr['prediccion'], name='Predicción', line=dict(color='#e63946',width=2,dash='dash')))
            fig.update_layout(title='Predicción vs Real (periodo de prueba)', title_font_color='#003366',
                            xaxis_title='Fecha', yaxis_title='Cantidad')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            fi = result['feature_importance']
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            st.subheader("Importancia de variables")
            st.bar_chart(fi_sorted)
        else:
            st.error(f"❌ {result['error']}")

with tab2:
    st.subheader(f"Predicción para los próximos {dias} días")
    if st.button("🔮 Generar Predicción", type="primary", use_container_width=True):
        with st.spinner("Generando predicción..."):
            result = predict_future(nit, df, producto, dias)
        
        if result['ok']:
            df_pred = pd.DataFrame(result['predicciones'])
            
            # Histórico + predicción
            hist = df[df['producto']==producto].groupby('fecha')['cantidad'].sum().reset_index()
            hist = hist.sort_values('fecha').tail(90)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['fecha'], y=hist['cantidad'], name='Histórico', line=dict(color='#003366',width=2)))
            fig.add_trace(go.Scatter(x=df_pred['fecha'], y=df_pred['prediccion'], name='Predicción', line=dict(color='#e63946',width=2,dash='dash')))
            fig.update_layout(title=f'Predicción de Demanda: {producto} ({dias} días)', title_font_color='#003366')
            st.plotly_chart(fig, use_container_width=True)
            
            col1,col2,col3 = st.columns(3)
            col1.metric("📦 Total estimado", f"{result['total']:.0f} uds")
            col2.metric("📊 Promedio diario", f"{result['total']/len(df_pred):.0f} uds/día")
            hist_avg = hist['cantidad'].tail(30).mean()
            cambio = ((result['total']/len(df_pred) - hist_avg)/hist_avg*100) if hist_avg>0 else 0
            col3.metric("📈 vs mes anterior", f"{cambio:+.1f}%")
            
            st.success(f"**Recomendación:** Comprar aprox. **{result['total']:.0f} unidades** de {producto} para los próximos {dias} días")
            
            with st.expander("📋 Detalle diario"):
                st.dataframe(df_pred, use_container_width=True)
        else:
            st.error(f"❌ {result['error']}")
