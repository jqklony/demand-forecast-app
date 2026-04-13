# -*- coding: utf-8 -*-
"""
Backtesting: Predecir sobre datos pasados y comparar con la realidad.
Muestra porcentaje de acierto 0-100%.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, json, joblib
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import list_empresas, load_consolidated_data, get_empresa_dir

st.set_page_config(page_title="Backtesting | Demand Forecast", page_icon="📊", layout="wide")
st.title("📊 Backtesting — Validación con Datos Históricos")

st.markdown("""
<div style='background:#f0f4f8;border-left:4px solid #003366;padding:15px;border-radius:0 8px 8px 0;margin:10px 0'>
<strong>¿Qué es el Backtesting?</strong><br>
El modelo predice sobre datos <strong>del pasado</strong> donde ya sabemos la respuesta real.
Luego comparamos predicción vs realidad para medir qué tan confiable es el modelo.
El resultado es un <strong>porcentaje de precisión del 0% al 100%</strong>.
</div>
""", unsafe_allow_html=True)

empresas = list_empresas()
if not empresas:
    st.warning("⚠️ Registre una empresa primero"); st.stop()

nit_options = {f"{e['nombre']} (NIT: {e['nit']})": e['nit'] for e in empresas}
selected = st.selectbox("Empresa:", list(nit_options.keys()))
nit = nit_options[selected]

df = load_consolidated_data(nit)
if df is None:
    st.warning("⚠️ No hay datos. Suba datos primero."); st.stop()

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    producto = st.selectbox("Producto:", sorted(df['producto'].unique()))
with col2:
    pct_test = st.slider("% de datos para validación:", 10, 40, 20, 5,
                         help="Se usa el último X% de datos como 'futuro' para validar")

with st.expander("❓ ¿Qué significa '% de datos para validación'? (click para ver explicación)"):
    st.markdown("""
    ### ¿Cómo funciona?
    
    Imagine que tiene **100 días** de datos históricos de ventas. Este slider controla cómo se dividen:
    
    | Slider | Entrenamiento | Validación | Resultado |
    |--------|--------------|------------|-----------|
    | **10%** | 90 días (pasado lejano) | 10 días (recientes) | Modelo más preciso, pero validación corta |
    | **20%** ⭐ Recomendado | 80 días | 20 días | **Balance óptimo** entre precisión y confiabilidad |
    | **30%** | 70 días | 30 días | Validación más confiable, modelo con menos datos |
    | **40%** | 60 días | 40 días | Mucha validación pero modelo puede perder precisión |
    
    **Visualmente:**
    ```
    Sus datos: [■■■■■■■■■■■■■■■■■■■■|■■■■■]
                ← El modelo aprende →  ← Se prueba aquí →
                   de estos datos       Predicción vs Realidad
    ```
    
    ### ¿Cuál elegir?
    - **20%** → Úselo por defecto. Es el estándar de la industria.
    - **30%** → Si quiere estar más seguro de que el resultado es confiable.
    - **10%** → Solo si tiene pocos datos (menos de 3 meses).
    - **40%** → Para una validación exhaustiva (pero el modelo tendrá menos datos para aprender).
    
    💡 **Consejo:** Pruebe con diferentes valores y compare los resultados. Si la precisión cambia 
    mucho entre 20% y 30%, el modelo puede ser inestable y necesitar más datos.
    """)

if st.button("🔍 Ejecutar Backtesting", type="primary", use_container_width=True):
    with st.spinner("Ejecutando backtesting..."):
        # Preparar serie temporal
        ts = df[df['producto'] == producto].groupby('fecha').agg(
            ventas=('cantidad', 'sum')
        ).reset_index().sort_values('fecha')
        
        if len(ts) < 60:
            st.error(f"Insuficientes datos: {len(ts)} días (mínimo 60)")
            st.stop()
        
        # Features
        ts['dia_semana'] = ts['fecha'].dt.dayofweek
        ts['mes'] = ts['fecha'].dt.month
        ts['dia_mes'] = ts['fecha'].dt.day
        ts['quincena'] = (ts['dia_mes'] <= 15).astype(int)
        ts['es_fin_mes'] = (ts['dia_mes'] >= 28).astype(int)
        for lag in [1, 7, 14, 30]:
            ts[f'lag_{lag}'] = ts['ventas'].shift(lag)
        for w in [7, 14, 30]:
            ts[f'mm_{w}'] = ts['ventas'].rolling(w, min_periods=1).mean()
        ts['tendencia'] = ts['mm_7'] - ts['mm_7'].shift(7)
        
        ts_clean = ts.dropna()
        features = ['dia_semana','mes','dia_mes','quincena','es_fin_mes',
                    'lag_1','lag_7','lag_14','lag_30','mm_7','mm_14','mm_30','tendencia']
        
        # Split temporal
        split_idx = int(len(ts_clean) * (1 - pct_test/100))
        train = ts_clean.iloc[:split_idx]
        test = ts_clean.iloc[split_idx:]
        
        X_train = train[features]; y_train = train['ventas']
        X_test = test[features]; y_test = test['ventas']
        
        # Entrenar XGBoost
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)
        
        # === MÉTRICAS ===
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        # MAPE
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask].values - pred[mask]) / y_test[mask].values)) * 100
        else:
            mape = 0
        
        # Porcentaje de precisión (100% - MAPE, acotado 0-100)
        precision_pct = max(0, min(100, 100 - mape))
        
        # Dirección correcta (sube/baja)
        if len(y_test) > 1:
            real_dir = np.diff(y_test.values) > 0
            pred_dir = np.diff(pred) > 0
            dir_accuracy = np.mean(real_dir == pred_dir) * 100
        else:
            dir_accuracy = 0
        
        # === DISPLAY ===
        st.markdown("## 📊 Resultados del Backtesting")
        
        # Gauge de precisión
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=precision_pct,
                title={'text': "Precisión del Modelo"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#003366"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#d4edda"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            fig_gauge2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dir_accuracy,
                title={'text': "Acierto de Tendencia"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2a9d8f"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 75], 'color': "#fff3cd"},
                        {'range': [75, 100], 'color': "#d4edda"},
                    ],
                }
            ))
            fig_gauge2.update_layout(height=300)
            st.plotly_chart(fig_gauge2, use_container_width=True)
        
        with col3:
            st.markdown("### Métricas detalladas")
            st.metric("MAE (Error promedio)", f"{mae:.1f} unidades")
            st.metric("MAPE (Error %)", f"{mape:.1f}%")
            st.metric("R² (Explicación)", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:.1f}")
        
        # === GRÁFICO PREDICCIÓN VS REAL ===
        st.markdown("### Predicción vs Realidad (datos históricos)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test['fecha'].values, y=y_test.values,
            name='Real (lo que pasó)', line=dict(color='#003366', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=test['fecha'].values, y=pred,
            name='Predicción del modelo', line=dict(color='#e63946', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f'Backtesting: {producto} — Precisión: {precision_pct:.0f}%',
            title_font_color='#003366',
            xaxis_title='Fecha', yaxis_title='Cantidad (unidades)',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # === TABLA DIARIA ===
        with st.expander("📋 Ver detalle día a día"):
            df_detail = pd.DataFrame({
                'Fecha': test['fecha'].dt.strftime('%Y-%m-%d').values,
                'Real': y_test.values.round(0).astype(int),
                'Predicción': pred.round(0).astype(int),
                'Error': np.abs(y_test.values - pred).round(1),
                'Error %': (np.abs(y_test.values - pred) / np.maximum(y_test.values, 1) * 100).round(1),
            })
            st.dataframe(df_detail, use_container_width=True, height=400)
        
        # === INTERPRETACIÓN ===
        st.markdown("### 💡 Interpretación de los resultados")
        
        if precision_pct >= 80:
            st.markdown(f"""
            <div style='background:#d4edda;border-left:4px solid #2a9d8f;padding:15px;border-radius:0 8px 8px 0'>
            <h4>🟢 EXCELENTE — Precisión: {precision_pct:.0f}%</h4>
            <p>El modelo es <strong>muy confiable</strong> para este producto. Las predicciones se desvían 
            en promedio solo <strong>{mae:.1f} unidades</strong> del valor real.</p>
            <p><strong>Recomendación:</strong> Puede usar las predicciones con confianza para planificar compras.</p>
            </div>
            """, unsafe_allow_html=True)
        elif precision_pct >= 60:
            st.markdown(f"""
            <div style='background:#fff3cd;border-left:4px solid #e9c46a;padding:15px;border-radius:0 8px 8px 0'>
            <h4>🟡 ACEPTABLE — Precisión: {precision_pct:.0f}%</h4>
            <p>El modelo tiene un rendimiento <strong>moderado</strong>. Se desvía en promedio 
            <strong>{mae:.1f} unidades</strong>. Esto es normal para productos con demanda irregular.</p>
            <p><strong>Recomendación:</strong> Use las predicciones como referencia pero ajuste con su experiencia.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#f8d7da;border-left:4px solid #e63946;padding:15px;border-radius:0 8px 8px 0'>
            <h4>🔴 BAJO — Precisión: {precision_pct:.0f}%</h4>
            <p>El modelo tiene dificultades con este producto. Posibles causas:</p>
            <ul>
            <li>Demanda muy irregular o con muchos picos impredecibles</li>
            <li>Pocos datos históricos (necesita más meses de datos)</li>
            <li>Factores externos no capturados (promociones, clima, eventos)</li>
            </ul>
            <p><strong>Recomendación:</strong> Recopile más datos o incluya variables adicionales.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Periodo analizado
        st.caption(f"Periodo de entrenamiento: {train['fecha'].min().strftime('%Y-%m-%d')} a {train['fecha'].max().strftime('%Y-%m-%d')} ({len(train)} días)")
        st.caption(f"Periodo de validación: {test['fecha'].min().strftime('%Y-%m-%d')} a {test['fecha'].max().strftime('%Y-%m-%d')} ({len(test)} días)")
