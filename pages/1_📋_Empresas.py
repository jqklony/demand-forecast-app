# -*- coding: utf-8 -*-
"""Página: Gestión de Empresas por NIT."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import save_empresa_info, list_empresas, load_empresa_info

st.set_page_config(page_title="Empresas | Demand Forecast", page_icon="🏢", layout="wide")
st.title("🏢 Gestión de Empresas")

tab1, tab2 = st.tabs(["📝 Registrar Empresa", "📋 Empresas Registradas"])

with tab1:
    st.subheader("Registrar nueva empresa")
    with st.form("form_empresa"):
        col1, col2 = st.columns(2)
        with col1:
            nit = st.text_input("NIT *", placeholder="900123456-7")
            nombre = st.text_input("Razón Social *", placeholder="Distribuidora XYZ S.A.S.")
        with col2:
            sector = st.selectbox("Sector *", [
                "Distribución Mayorista", "Retail / Tienda", "Alimentos y Bebidas",
                "Materiales de Construcción", "Productos de Aseo", "Farmacia",
                "Tecnología", "Otro"
            ])
            contacto = st.text_input("Contacto", placeholder="email@empresa.com")
        
        submitted = st.form_submit_button("Registrar Empresa", type="primary", use_container_width=True)
        if submitted:
            if not nit or not nombre:
                st.error("NIT y Razón Social son obligatorios")
            else:
                info = save_empresa_info(nit, nombre, sector, contacto)
                st.success(f"✅ Empresa **{nombre}** (NIT: {nit}) registrada exitosamente")
                st.balloons()

with tab2:
    empresas = list_empresas()
    if not empresas:
        st.info("No hay empresas registradas. Registre una en la pestaña anterior.")
    else:
        st.subheader(f"{len(empresas)} empresas registradas")
        for emp in empresas:
            with st.expander(f"🏢 {emp.get('nombre','N/A')} — NIT: {emp.get('nit','N/A')}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Sector", emp.get('sector','N/A'))
                col2.metric("Registros", emp.get('total_registros',0))
                col3.metric("Productos", emp.get('productos','N/A'))
                if emp.get('periodo_inicio'):
                    st.caption(f"Periodo: {emp.get('periodo_inicio','?')} a {emp.get('periodo_fin','?')}")
                if emp.get('ultimo_upload'):
                    st.caption(f"Último upload: {emp['ultimo_upload'][:19]}")
