# -*- coding: utf-8 -*-
"""Gestión de datos por empresa (NIT)."""
import os, json, hashlib
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'empresas')

def get_empresa_dir(nit: str) -> str:
    """Obtener o crear directorio de una empresa por NIT."""
    safe_nit = nit.replace('-','').replace('.','').strip()
    path = os.path.join(DATA_DIR, safe_nit)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(path, 'modelos'), exist_ok=True)
    os.makedirs(os.path.join(path, 'reportes'), exist_ok=True)
    return path

def save_empresa_info(nit: str, nombre: str, sector: str, contacto: str = ""):
    """Guardar información de la empresa."""
    path = get_empresa_dir(nit)
    info = {
        'nit': nit, 'nombre': nombre, 'sector': sector,
        'contacto': contacto, 'fecha_registro': datetime.now().isoformat(),
        'ultimo_upload': None, 'total_registros': 0
    }
    info_path = os.path.join(path, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            existing = json.load(f)
        existing.update({k:v for k,v in info.items() if v})
        info = existing
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info

def load_empresa_info(nit: str) -> dict:
    """Cargar información de la empresa."""
    path = get_empresa_dir(nit)
    info_path = os.path.join(path, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def list_empresas() -> list:
    """Listar todas las empresas registradas."""
    empresas = []
    if os.path.exists(DATA_DIR):
        for d in os.listdir(DATA_DIR):
            info_path = os.path.join(DATA_DIR, d, 'info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    empresas.append(json.load(f))
    return empresas

def upload_data(nit: str, df: pd.DataFrame, filename: str) -> dict:
    """Subir y validar datos de ventas de una empresa."""
    path = get_empresa_dir(nit)
    
    # Validar columnas mínimas
    required = ['fecha']
    optional_sets = [
        ['producto', 'cantidad'],  # mínimo
        ['producto', 'cantidad', 'valor_total'],  # ideal
        ['producto', 'cantidad', 'valor_total', 'cliente'],  # completo
    ]
    
    cols_lower = [c.lower().strip() for c in df.columns]
    df.columns = cols_lower
    
    has_fecha = 'fecha' in cols_lower
    has_prod = 'producto' in cols_lower
    has_cant = 'cantidad' in cols_lower
    
    if not has_fecha:
        return {'ok': False, 'error': 'Falta columna "fecha"'}
    if not has_prod:
        return {'ok': False, 'error': 'Falta columna "producto"'}
    if not has_cant:
        return {'ok': False, 'error': 'Falta columna "cantidad"'}
    
    # Parse fecha
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    
    if len(df) == 0:
        return {'ok': False, 'error': 'No hay registros válidos después de parsear fechas'}
    
    # Guardar archivo
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"upload_{ts}_{filename}"
    save_path = os.path.join(path, 'uploads', save_name)
    df.to_csv(save_path, index=False)
    
    # Actualizar datos consolidados
    consolidated_path = os.path.join(path, 'datos_consolidados.csv')
    if os.path.exists(consolidated_path):
        existing = pd.read_csv(consolidated_path, parse_dates=['fecha'])
        df_all = pd.concat([existing, df], ignore_index=True)
        df_all = df_all.drop_duplicates()
    else:
        df_all = df
    
    df_all.to_csv(consolidated_path, index=False)
    
    # Actualizar info
    info = load_empresa_info(nit) or {}
    info['ultimo_upload'] = datetime.now().isoformat()
    info['total_registros'] = len(df_all)
    info['productos'] = int(df_all['producto'].nunique())
    info['periodo_inicio'] = str(df_all['fecha'].min())
    info['periodo_fin'] = str(df_all['fecha'].max())
    info_path = os.path.join(path, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    return {
        'ok': True,
        'registros_nuevos': len(df),
        'total_consolidado': len(df_all),
        'productos': int(df_all['producto'].nunique()),
        'periodo': f"{df_all['fecha'].min().strftime('%Y-%m-%d')} a {df_all['fecha'].max().strftime('%Y-%m-%d')}"
    }

def load_consolidated_data(nit: str) -> pd.DataFrame:
    """Cargar datos consolidados de una empresa."""
    path = get_empresa_dir(nit)
    consolidated_path = os.path.join(path, 'datos_consolidados.csv')
    if os.path.exists(consolidated_path):
        return pd.read_csv(consolidated_path, parse_dates=['fecha'])
    return None
