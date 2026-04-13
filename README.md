# 📊 Demand Forecast System

Sistema de Predicción de Demanda con Machine Learning para Distribuidores Mayoristas y PYMES.

## 🚀 Características

- **Multi-empresa:** Gestión por NIT, cada empresa tiene sus datos y modelos independientes
- **Upload de datos:** Suba archivos Excel/CSV con historial de ventas
- **Predicción:** Modelos XGBoost que predicen demanda por producto a 7-90 días
- **Testing:** Monitoreo de salud del modelo, detección de degradación automática
- **Reentrenamiento:** Actualice el modelo mensualmente con datos nuevos

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## 🏃 Ejecución local

```bash
streamlit run app.py
```

Abre en: http://localhost:8501

## 🌐 Despliegue en Streamlit Cloud

1. Suba este repo a GitHub
2. Vaya a https://share.streamlit.io
3. Conecte su repo → se despliega automáticamente

## 📎 Formato de datos

| fecha | producto | cantidad | valor_total | cliente |
|-------|----------|----------|-------------|---------|
| 2024-01-05 | Arroz 50kg | 24 | 1440000 | CLI-001 |

**Obligatorias:** `fecha`, `producto`, `cantidad`
**Opcionales:** `valor_total`, `cliente`, `en_promocion`

## 🛠️ Tecnologías

- Python 3.10+
- Streamlit (dashboard)
- XGBoost (modelo predictivo)
- Pandas / NumPy (procesamiento)
- Plotly (visualización)
- scikit-learn (evaluación)

## 📄 Licencia

Trabajo de Especialización - 2026
