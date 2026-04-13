# -*- coding: utf-8 -*-
"""Motor de modelos: entrenamiento, predicción, testing y reentrenamiento."""
import os, json, time, joblib
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelos')

def _build_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Construir features temporales para la serie."""
    df = ts.copy()
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['mes'] = df['fecha'].dt.month
    df['dia_mes'] = df['fecha'].dt.day
    df['quincena'] = (df['dia_mes'] <= 15).astype(int)
    df['es_fin_mes'] = (df['dia_mes'] >= 28).astype(int)
    df['semana_ano'] = df['fecha'].dt.isocalendar().week.astype(int)
    
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df['ventas'].shift(lag)
    for w in [7, 14, 30]:
        df[f'mm_{w}'] = df['ventas'].rolling(w, min_periods=1).mean()
    df['tendencia'] = df['mm_7'] - df['mm_7'].shift(7)
    
    return df.dropna()

FEATURE_COLS = ['dia_semana','mes','dia_mes','quincena','es_fin_mes',
                'lag_1','lag_7','lag_14','lag_30','mm_7','mm_14','mm_30','tendencia']

def train_model(nit: str, df: pd.DataFrame, producto: str, test_ratio: float = 0.2) -> dict:
    """Entrenar modelo para un producto de una empresa."""
    from utils.data_manager import get_empresa_dir
    
    # Agregar por día
    ts = df[df['producto'] == producto].groupby('fecha').agg(
        ventas=('cantidad', 'sum')
    ).reset_index().sort_values('fecha')
    
    if len(ts) < 60:
        return {'ok': False, 'error': f'Insuficientes datos: {len(ts)} días (mínimo 60)'}
    
    ts_feat = _build_features(ts)
    
    # Split temporal
    split_idx = int(len(ts_feat) * (1 - test_ratio))
    train = ts_feat.iloc[:split_idx]
    test = ts_feat.iloc[split_idx:]
    
    X_train = train[FEATURE_COLS]
    y_train = train['ventas']
    X_test = test[FEATURE_COLS]
    y_test = test['ventas']
    
    # Entrenar XGBoost
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - t0
    
    # Evaluar
    pred = model.predict(X_test)
    pred = np.maximum(pred, 0)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    mape = 0
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask].values - pred[mask]) / y_test[mask].values)) * 100
    
    # Guardar modelo
    empresa_dir = get_empresa_dir(nit)
    model_path = os.path.join(empresa_dir, 'modelos', f'model_{producto.replace(" ","_")}.pkl')
    joblib.dump(model, model_path)
    
    # Guardar metadata
    meta = {
        'producto': producto,
        'fecha_entrenamiento': datetime.now().isoformat(),
        'registros_train': len(train),
        'registros_test': len(test),
        'metricas': {'MAE': round(mae,2), 'RMSE': round(rmse,2), 'R2': round(r2,4), 'MAPE': round(mape,2)},
        'tiempo_entrenamiento': f"{train_time:.1f}s",
        'features': FEATURE_COLS,
        'periodo_train': f"{train['fecha'].min().strftime('%Y-%m-%d')} a {train['fecha'].max().strftime('%Y-%m-%d')}",
        'periodo_test': f"{test['fecha'].min().strftime('%Y-%m-%d')} a {test['fecha'].max().strftime('%Y-%m-%d')}",
    }
    meta_path = os.path.join(empresa_dir, 'modelos', f'meta_{producto.replace(" ","_")}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    return {
        'ok': True,
        'metricas': meta['metricas'],
        'train_size': len(train),
        'test_size': len(test),
        'pred_vs_real': {'fechas': test['fecha'].dt.strftime('%Y-%m-%d').tolist(),
                         'real': y_test.tolist(), 'prediccion': pred.tolist()},
        'feature_importance': dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
    }

def predict_future(nit: str, df: pd.DataFrame, producto: str, dias: int = 30) -> dict:
    """Generar predicción futura."""
    from utils.data_manager import get_empresa_dir
    empresa_dir = get_empresa_dir(nit)
    model_path = os.path.join(empresa_dir, 'modelos', f'model_{producto.replace(" ","_")}.pkl')
    
    if not os.path.exists(model_path):
        return {'ok': False, 'error': 'Modelo no encontrado. Entrene primero.'}
    
    model = joblib.load(model_path)
    
    ts = df[df['producto'] == producto].groupby('fecha').agg(
        ventas=('cantidad', 'sum')
    ).reset_index().sort_values('fecha')
    
    ts_feat = _build_features(ts)
    last_data = ts_feat.tail(30).copy()
    last_date = ts_feat['fecha'].max()
    
    predictions = []
    for i in range(dias):
        fd = last_date + timedelta(days=i+1)
        if fd.dayofweek >= 6: continue  # skip domingo
        
        row = {
            'dia_semana': fd.dayofweek, 'mes': fd.month, 'dia_mes': fd.day,
            'quincena': int(fd.day <= 15), 'es_fin_mes': int(fd.day >= 28),
            'lag_1': last_data['ventas'].iloc[-1],
            'lag_7': last_data['ventas'].iloc[-7] if len(last_data)>=7 else last_data['ventas'].mean(),
            'lag_14': last_data['ventas'].iloc[-14] if len(last_data)>=14 else last_data['ventas'].mean(),
            'lag_30': last_data['ventas'].mean(),
            'mm_7': last_data['ventas'].tail(7).mean(),
            'mm_14': last_data['ventas'].tail(14).mean(),
            'mm_30': last_data['ventas'].mean(),
            'tendencia': last_data['ventas'].tail(7).mean() - last_data['ventas'].tail(14).mean(),
        }
        
        pred_val = max(0, float(model.predict(pd.DataFrame([row]))[0]))
        predictions.append({'fecha': fd.strftime('%Y-%m-%d'), 'prediccion': round(pred_val, 1)})
        
        new_row = pd.DataFrame([{'fecha': fd, 'ventas': pred_val}])
        last_data = pd.concat([last_data, new_row], ignore_index=True)
    
    return {'ok': True, 'predicciones': predictions, 'total': sum(p['prediccion'] for p in predictions)}

def test_model_health(nit: str, df: pd.DataFrame, producto: str) -> dict:
    """Testing de salud del modelo: detectar degradación."""
    from utils.data_manager import get_empresa_dir
    empresa_dir = get_empresa_dir(nit)
    meta_path = os.path.join(empresa_dir, 'modelos', f'meta_{producto.replace(" ","_")}.json')
    model_path = os.path.join(empresa_dir, 'modelos', f'model_{producto.replace(" ","_")}.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {'ok': False, 'error': 'Modelo no encontrado'}
    
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Tomar datos recientes (último mes)
    ts = df[df['producto'] == producto].groupby('fecha').agg(
        ventas=('cantidad', 'sum')
    ).reset_index().sort_values('fecha')
    
    ts_feat = _build_features(ts)
    recent = ts_feat.tail(30)  # último mes
    
    if len(recent) < 10:
        return {'ok': False, 'error': 'Insuficientes datos recientes para testing'}
    
    X_recent = recent[FEATURE_COLS]
    y_recent = recent['ventas']
    pred_recent = model.predict(X_recent)
    pred_recent = np.maximum(pred_recent, 0)
    
    mae_current = mean_absolute_error(y_recent, pred_recent)
    mae_original = meta['metricas']['MAE']
    
    mape_current = 0
    mask = y_recent != 0
    if mask.sum() > 0:
        mape_current = np.mean(np.abs((y_recent[mask].values - pred_recent[mask]) / y_recent[mask].values)) * 100
    
    degradation = ((mae_current - mae_original) / mae_original) * 100 if mae_original > 0 else 0
    
    health = 'BUENO'
    if degradation > 20: health = 'DEGRADADO'
    if degradation > 50: health = 'CRITICO'
    
    needs_retrain = degradation > 20
    
    return {
        'ok': True,
        'salud': health,
        'mae_original': mae_original,
        'mae_actual': round(mae_current, 2),
        'mape_actual': round(mape_current, 2),
        'degradacion_pct': round(degradation, 1),
        'necesita_reentrenamiento': needs_retrain,
        'fecha_entrenamiento': meta['fecha_entrenamiento'],
        'dias_desde_entrenamiento': (datetime.now() - datetime.fromisoformat(meta['fecha_entrenamiento'])).days,
    }

def retrain_with_new_data(nit: str, df: pd.DataFrame, producto: str) -> dict:
    """Reentrenar modelo con todos los datos actuales."""
    result = train_model(nit, df, producto, test_ratio=0.15)
    if result['ok']:
        result['mensaje'] = 'Modelo reentrenado exitosamente con datos actualizados'
    return result
