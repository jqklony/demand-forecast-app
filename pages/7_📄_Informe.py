# -*- coding: utf-8 -*-
"""
Generador de Informe de Predicción en español.
Genera un reporte descargable con los resultados.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, load_consolidated_data, get_empresa_dir, load_empresa_info
from utils.model_engine import predict_future

st.set_page_config(page_title="Informe | Demand Forecast", page_icon="📄", layout="wide")
st.title("📄 Generador de Informe de Predicción")

st.markdown("""
Genere un informe detallado y descargable con las predicciones de demanda, 
métricas del modelo y recomendaciones de compra.
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

info = load_empresa_info(nit) or {}

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    productos_sel = st.multiselect("Productos a incluir:", sorted(df['producto'].unique()),
                                    default=sorted(df['producto'].unique())[:5])
with col2:
    dias = st.slider("Días a predecir:", 7, 90, 30)

if st.button("📄 Generar Informe Completo", type="primary", use_container_width=True):
    with st.spinner("Generando informe con predicciones para cada producto..."):
        
        fecha_hoy = datetime.now().strftime('%d de %B de %Y')
        
        # Header del informe
        informe = f"""# INFORME DE PREDICCIÓN DE DEMANDA

---

**Empresa:** {info.get('nombre', selected)}  
**NIT:** {nit}  
**Sector:** {info.get('sector', 'No especificado')}  
**Fecha del informe:** {fecha_hoy}  
**Periodo de datos:** {info.get('periodo_inicio', 'N/A')} a {info.get('periodo_fin', 'N/A')}  
**Total de registros:** {info.get('total_registros', len(df)):,}  
**Horizonte de predicción:** {dias} días  

---

## RESUMEN EJECUTIVO

Este informe presenta las predicciones de demanda para los próximos {dias} días 
generadas por un modelo de Machine Learning (XGBoost) entrenado con los datos 
históricos de ventas de la empresa. Las predicciones permiten planificar compras 
y optimizar niveles de inventario.

---

## PREDICCIONES POR PRODUCTO

"""
        total_general = 0
        tabla_resumen = []
        
        for prod in productos_sel:
            result = predict_future(nit, df, prod, dias)
            
            if result['ok']:
                preds = result['predicciones']
                total = result['total']
                total_general += total
                prom_diario = total / len(preds) if preds else 0
                
                # Histórico
                hist = df[df['producto'] == prod].groupby('fecha')['cantidad'].sum()
                hist_prom = hist.tail(30).mean() if len(hist) > 0 else 0
                cambio = ((prom_diario - hist_prom) / hist_prom * 100) if hist_prom > 0 else 0
                
                tabla_resumen.append({
                    'Producto': prod,
                    'Total estimado': f"{total:.0f} uds",
                    'Promedio diario': f"{prom_diario:.0f} uds/día",
                    'Prom. mes anterior': f"{hist_prom:.0f} uds/día",
                    'Variación': f"{cambio:+.1f}%",
                    'Tendencia': '📈 Subiendo' if cambio > 5 else ('📉 Bajando' if cambio < -5 else '➡️ Estable')
                })
                
                informe += f"""### {prod}

- **Total estimado para {dias} días:** {total:.0f} unidades
- **Promedio diario estimado:** {prom_diario:.0f} unidades/día
- **Promedio del mes anterior:** {hist_prom:.0f} unidades/día
- **Variación:** {cambio:+.1f}% {'(AUMENTO)' if cambio > 0 else '(DISMINUCIÓN)' if cambio < 0 else '(ESTABLE)'}
- **Recomendación de compra:** {total:.0f} unidades

"""
            else:
                informe += f"""### {prod}

⚠️ No se pudo generar predicción: {result.get('error', 'Modelo no entrenado')}
Entrene el modelo primero en la sección 🔮 Predicción.

"""
        
        informe += f"""---

## RESUMEN DE COMPRAS RECOMENDADAS

**Total general estimado para {dias} días:** {total_general:,.0f} unidades en todos los productos.

"""
        
        informe += """---

## METODOLOGÍA

### Modelo utilizado: XGBoost (Extreme Gradient Boosting)

XGBoost es un algoritmo de Machine Learning que construye cientos de árboles de decisión 
de forma secuencial, donde cada árbol corrige los errores del anterior. Es el algoritmo 
más utilizado en la industria para predicción de demanda por su:

- **Alta precisión** en datos tabulares con variables temporales
- **Capacidad de capturar estacionalidad** (semanal, quincenal, mensual, anual)
- **Robustez ante datos faltantes** y outliers
- **Velocidad** de entrenamiento y predicción

### Variables utilizadas por el modelo

| Variable | Descripción |
|----------|-------------|
| Día de la semana | Patrón semanal (lunes diferente de sábado) |
| Mes del año | Estacionalidad anual (diciembre = pico) |
| Quincena | Efecto nómina (2da quincena = más ventas) |
| Fin de mes | Cierre de mes = más pedidos |
| Ventas pasadas (lag) | Ventas de hace 1, 7, 14 y 30 días |
| Medias móviles | Promedio de últimos 7, 14 y 30 días |
| Tendencia | ¿Las ventas van subiendo o bajando? |

### Limitaciones

- Las predicciones no consideran eventos imprevistos (crisis, pandemias, desastres)
- Promociones futuras no incluidas (se recomienda agregar esta variable)
- La precisión mejora con más datos históricos (mínimo 1 año recomendado)

---

## NOTAS

- Este informe fue generado automáticamente por el Sistema de Predicción de Demanda.
- Las predicciones son estimaciones basadas en patrones históricos y deben complementarse 
  con el criterio del equipo de compras.
- Se recomienda reentrenar el modelo mensualmente con datos actualizados.
- Para validar la precisión del modelo, use el módulo de Backtesting.

---

*Generado por Demand Forecast System v1.0 | Machine Learning + XGBoost*
"""
        
        # Mostrar informe
        st.markdown("---")
        st.markdown("## Vista previa del informe")
        st.markdown(informe)
        
        # Tabla resumen
        if tabla_resumen:
            st.markdown("### Tabla resumen de predicciones")
            st.dataframe(pd.DataFrame(tabla_resumen), use_container_width=True)
        
        # Botón de descarga
        st.markdown("---")
        st.download_button(
            label="📥 Descargar informe como texto",
            data=informe,
            file_name=f"Informe_Prediccion_{nit}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )
        
        # También descarga CSV de predicciones
        if tabla_resumen:
            csv = pd.DataFrame(tabla_resumen).to_csv(index=False)
            st.download_button(
                label="📥 Descargar tabla de predicciones (CSV)",
                data=csv,
                file_name=f"Predicciones_{nit}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
