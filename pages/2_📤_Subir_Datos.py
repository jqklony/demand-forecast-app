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
**Idioma:** Las columnas pueden estar en español o inglés
""")

with st.expander("📋 Ver todas las columnas aceptadas (obligatorias y opcionales)", expanded=True):
    st.markdown("#### ✅ Columnas obligatorias (mínimo necesario)")
    st.markdown("""
    | Columna | Descripción | Ejemplo |
    |---------|-------------|---------|
    | `fecha` | Fecha de la venta o pedido | 2024-03-15 |
    | `producto` | Nombre o código del producto | Arroz 50kg |
    | `cantidad` | Unidades vendidas o pedidas | 24 |
    """)
    
    st.markdown("#### 📌 Columnas opcionales (mejoran la precisión del modelo)")
    st.markdown("""
    | Columna | Descripción | Ejemplo | ¿Qué aporta al modelo? |
    |---------|-------------|---------|----------------------|
    | `valor_total` | Valor monetario de la venta (COP) | 1440000 | Permite analizar ingresos y valor promedio por pedido |
    | `cliente` | ID o nombre del cliente | CLI-001 | Permite predecir demanda **por cliente** (útil para mayoristas B2B) |
    | `en_promocion` | Si el producto estaba en promoción (1=Sí, 0=No) | 1 | Las promociones aumentan la demanda 20-40%. Sin esta columna, el modelo no distingue ventas normales de promocionales |
    | `categoria` | Categoría o familia del producto | Granos, Bebidas, Aseo | Permite agrupar productos similares y detectar patrones por categoría |
    | `precio_unitario` | Precio por unidad del producto | 60000 | Permite analizar elasticidad precio-demanda (si sube el precio, ¿baja la demanda?) |
    | `vendedor` | Vendedor o asesor comercial | V-005 | Identifica patrones de venta por vendedor |
    | `ciudad` | Ciudad o zona de la venta | Bogotá, Medellín | Permite predicciones segmentadas por región geográfica |
    | `canal` | Canal de venta | Presencial, Online, WhatsApp | Diferencia comportamiento por canal de distribución |
    | `bodega` | Bodega o punto de despacho | Bodega Norte, Bodega Sur | Permite predicción por punto de almacenamiento |
    | `temperatura` | Temperatura del día (°C) | 28 | Impacta ventas de bebidas, helados, etc. Datos de IDEAM o Open-Meteo |
    | `dia_festivo` | Si fue día festivo (1=Sí, 0=No) | 1 | Los festivos cambian drásticamente los patrones de venta |
    | `descuento_pct` | Porcentaje de descuento aplicado | 15 | Cuantifica el impacto exacto del descuento en la demanda |
    | `inventario_inicial` | Stock disponible al inicio del día | 150 | Permite detectar ventas perdidas por desabastecimiento |
    | `pedidos_pendientes` | Pedidos no despachados (backorders) | 5 | Indica demanda real no satisfecha |
    """)
    
    st.markdown("#### 💡 Recomendación según su tipo de negocio")
    st.markdown("""
    | Tipo de negocio | Columnas recomendadas además de las obligatorias |
    |-----------------|-----------------------------------------------|
    | **Distribuidor mayorista** | `cliente`, `valor_total`, `categoria`, `vendedor`, `ciudad` |
    | **Tienda / Retail** | `valor_total`, `en_promocion`, `categoria`, `precio_unitario` |
    | **E-commerce** | `cliente`, `valor_total`, `canal`, `en_promocion`, `ciudad` |
    | **Restaurante / Food** | `valor_total`, `temperatura`, `dia_festivo` |
    | **Cualquier negocio (mínimo)** | `valor_total`, `en_promocion` |
    
    ⚠️ **Entre más columnas proporcione, más preciso será el modelo.** Pero las 3 obligatorias son suficientes para empezar.
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
