# -*- coding: utf-8 -*-
"""
Gestión de Google Sheets para guardar predicciones e informes.
Requiere credenciales de cuenta de servicio de Google Cloud.
"""
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import json
import os
import streamlit as st
from datetime import datetime

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def get_gspread_client():
    """Obtener cliente de Google Sheets usando Streamlit secrets."""
    try:
        # Opción 1: Desde Streamlit secrets (para deploy en Streamlit Cloud)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            creds_dict = dict(st.secrets['gcp_service_account'])
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            return gspread.authorize(creds)
        
        # Opción 2: Desde archivo JSON local (para desarrollo)
        creds_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'credentials', 'service_account.json')
        if os.path.exists(creds_file):
            creds = Credentials.from_service_account_file(creds_file, scopes=SCOPES)
            return gspread.authorize(creds)
        
        return None
    except Exception as e:
        st.warning(f"⚠️ Google Sheets no configurado: {e}")
        return None

def get_or_create_spreadsheet(client, spreadsheet_name="DemandForecast_Historial"):
    """Obtener o crear la hoja de cálculo."""
    try:
        return client.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        sh = client.create(spreadsheet_name)
        # Compartir con el usuario (hacer pública para lectura)
        sh.share('', perm_type='anyone', role='reader')
        return sh

def get_or_create_worksheet(spreadsheet, nit, ws_type="predicciones"):
    """Obtener o crear pestaña para una empresa."""
    ws_name = f"{ws_type}_{nit}"
    try:
        return spreadsheet.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=ws_name, rows=1000, cols=20)
        # Headers según tipo
        if ws_type == "predicciones":
            headers = ['Fecha_Informe', 'NIT', 'Empresa', 'Producto', 'Dias_Prediccion',
                      'Total_Predicho', 'Promedio_Diario', 'Variacion_Pct', 'Tendencia',
                      'Precision_Modelo', 'Modelo_Usado']
        elif ws_type == "informes":
            headers = ['Fecha_Informe', 'NIT', 'Empresa', 'Tipo_Informe', 'Productos_Incluidos',
                      'Total_General', 'Horizonte_Dias', 'Resumen']
        elif ws_type == "backtesting":
            headers = ['Fecha_Test', 'NIT', 'Empresa', 'Producto', 'Pct_Validacion',
                      'Precision_Pct', 'MAE', 'MAPE', 'R2', 'Tendencia_Accuracy',
                      'Dias_Train', 'Dias_Test']
        else:
            headers = ['Fecha', 'Dato']
        ws.update('A1', [headers])
        # Formato header
        ws.format('A1:K1', {'backgroundColor': {'red':0,'green':0.2,'blue':0.4},
                            'textFormat': {'foregroundColor':{'red':1,'green':1,'blue':1},'bold':True}})
        return ws

def guardar_prediccion(nit, empresa_nombre, producto, dias, total, promedio, variacion, tendencia, precision=None):
    """Guardar una predicción en Google Sheets."""
    client = get_gspread_client()
    if client is None:
        return False, "Google Sheets no configurado"
    
    try:
        sh = get_or_create_spreadsheet(client)
        ws = get_or_create_worksheet(sh, nit, "predicciones")
        
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            nit, empresa_nombre, producto, dias,
            round(total, 0), round(promedio, 1), round(variacion, 1), tendencia,
            round(precision, 1) if precision else "N/A",
            "XGBoost"
        ]
        ws.append_row(row)
        return True, sh.url
    except Exception as e:
        return False, str(e)

def guardar_informe(nit, empresa_nombre, tipo, productos, total_general, horizonte, resumen):
    """Guardar un informe generado en Google Sheets."""
    client = get_gspread_client()
    if client is None:
        return False, "Google Sheets no configurado"
    
    try:
        sh = get_or_create_spreadsheet(client)
        ws = get_or_create_worksheet(sh, nit, "informes")
        
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            nit, empresa_nombre, tipo, ', '.join(productos) if isinstance(productos, list) else str(productos),
            round(total_general, 0), horizonte, resumen[:500]
        ]
        ws.append_row(row)
        return True, sh.url
    except Exception as e:
        return False, str(e)

def guardar_backtesting(nit, empresa_nombre, producto, pct_val, precision, mae, mape, r2, dir_acc, dias_train, dias_test):
    """Guardar resultado de backtesting."""
    client = get_gspread_client()
    if client is None:
        return False, "Google Sheets no configurado"
    
    try:
        sh = get_or_create_spreadsheet(client)
        ws = get_or_create_worksheet(sh, nit, "backtesting")
        
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            nit, empresa_nombre, producto, pct_val,
            round(precision, 1), round(mae, 2), round(mape, 1), round(r2, 4), round(dir_acc, 1),
            dias_train, dias_test
        ]
        ws.append_row(row)
        return True, sh.url
    except Exception as e:
        return False, str(e)

def cargar_historial(nit, ws_type="predicciones"):
    """Cargar historial de una empresa desde Google Sheets."""
    client = get_gspread_client()
    if client is None:
        return None
    
    try:
        sh = get_or_create_spreadsheet(client)
        ws = get_or_create_worksheet(sh, nit, ws_type)
        data = ws.get_all_records()
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        return None

def sheets_disponible():
    """Verificar si Google Sheets está configurado."""
    return get_gspread_client() is not None
