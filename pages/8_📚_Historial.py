# -*- coding: utf-8 -*-
"""Historial de predicciones, informes y backtesting guardados en Google Sheets."""
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas
from utils.sheets_manager import cargar_historial, sheets_disponible

st.set_page_config(page_title="Historial | Demand Forecast", page_icon="📚", layout="wide")
st.title("📚 Historial de Predicciones e Informes")

if not sheets_disponible():
    st.warning("""
    ⚠️ **Google Sheets no está configurado todavía.**
    
    Para habilitar el guardado automático de predicciones e informes, necesita:
    
    1. Crear un proyecto en [Google Cloud Console](https://console.cloud.google.com/)
    2. Habilitar las APIs de Google Sheets y Google Drive
    3. Crear una Cuenta de Servicio y descargar el JSON
    4. Agregar las credenciales en Streamlit Cloud → Settings → Secrets
    
    **Formato del secret en Streamlit Cloud:**
    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "su-proyecto"
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "cuenta@su-proyecto.iam.gserviceaccount.com"
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    ```
    
    Mientras tanto, las predicciones se pueden descargar manualmente con los botones de descarga.
    """)
    st.stop()

empresas = list_empresas()
if not empresas:
    st.info("No hay empresas registradas."); st.stop()

nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Empresa:", list(nit_options.keys()))
nit = nit_options[selected]

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predicciones", "📄 Informes", "📊 Backtesting"])

with tab1:
    st.subheader("🔮 Historial de Predicciones")
    df_pred = cargar_historial(nit, "predicciones")
    if df_pred is not None and len(df_pred) > 0:
        st.dataframe(df_pred, use_container_width=True)
        st.metric("Total predicciones guardadas", len(df_pred))
        
        csv = df_pred.to_csv(index=False)
        st.download_button("📥 Descargar historial completo (CSV)", csv, 
                          f"historial_predicciones_{nit}.csv", "text/csv", use_container_width=True)
    else:
        st.info("No hay predicciones guardadas para esta empresa. Genere predicciones en la pestaña 🔮 Predicción.")

with tab2:
    st.subheader("📄 Historial de Informes")
    df_inf = cargar_historial(nit, "informes")
    if df_inf is not None and len(df_inf) > 0:
        st.dataframe(df_inf, use_container_width=True)
        st.metric("Total informes guardados", len(df_inf))
    else:
        st.info("No hay informes guardados. Genere informes en la pestaña 📄 Informe.")

with tab3:
    st.subheader("📊 Historial de Backtesting")
    df_bt = cargar_historial(nit, "backtesting")
    if df_bt is not None and len(df_bt) > 0:
        st.dataframe(df_bt, use_container_width=True)
        
        # Gráfico de evolución de precisión
        if 'Precision_Pct' in df_bt.columns and len(df_bt) > 1:
            import plotly.express as px
            fig = px.line(df_bt, x='Fecha_Test', y='Precision_Pct', color='Producto',
                        title='Evolución de Precisión del Modelo por Producto',
                        labels={'Precision_Pct':'Precisión (%)','Fecha_Test':'Fecha'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay resultados de backtesting. Ejecute backtesting en la pestaña 📊 Backtesting.")
