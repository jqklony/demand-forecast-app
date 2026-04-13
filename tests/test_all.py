# -*- coding: utf-8 -*-
"""Tests completos del sistema Demand Forecast Pro."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils.data_manager import save_empresa_info, upload_data, load_consolidated_data, list_empresas
from utils.model_engine import train_model, predict_future, test_model_health, retrain_with_new_data

def test_all():
    print("=" * 55)
    print("  TESTS: Demand Forecast Pro")
    print("=" * 55)
    passed = 0
    total = 0

    # Test 1: Registrar empresa
    total += 1
    print("\nTest 1: Registrar empresa...")
    try:
        save_empresa_info('900123456', 'Distribuidora Demo SAS', 'Alimentos', 'test@demo.com')
        empresas = list_empresas()
        assert len(empresas) > 0
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 2: Subir datos
    total += 1
    print("\nTest 2: Subir datos de ventas...")
    try:
        np.random.seed(42)
        fechas = pd.date_range('2024-01-01', '2025-12-31', freq='B')
        records = []
        for f in fechas:
            for prod in ['Arroz 50kg', 'Aceite caja x12', 'Azucar 50kg']:
                qty = max(1, int(np.random.normal(20, 5) * (1.3 if f.month == 12 else 1.0)))
                records.append({
                    'fecha': f, 'producto': prod, 'cantidad': qty,
                    'valor_total': qty * 50000, 'cliente': 'CLI-001'
                })
        df_demo = pd.DataFrame(records)
        result = upload_data('900123456', df_demo, 'demo.csv')
        assert result['ok']
        print(f"  Registros: {result['registros_nuevos']:,}")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 3: Cargar datos consolidados
    total += 1
    print("\nTest 3: Cargar datos consolidados...")
    try:
        df = load_consolidated_data('900123456')
        assert df is not None
        assert len(df) > 1000
        print(f"  Registros: {len(df):,}, Productos: {df['producto'].nunique()}")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 4: Entrenar modelo
    total += 1
    print("\nTest 4: Entrenar modelo XGBoost...")
    try:
        result = train_model('900123456', df, 'Arroz 50kg')
        assert result['ok']
        m = result['metricas']
        print(f"  MAE={m['MAE']:.1f}, MAPE={m['MAPE']:.1f}%, R2={m['R2']:.3f}")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 5: Prediccion futura
    total += 1
    print("\nTest 5: Prediccion 30 dias...")
    try:
        result = predict_future('900123456', df, 'Arroz 50kg', 30)
        assert result['ok']
        assert result['total'] > 0
        print(f"  Total predicho: {result['total']:.0f} unidades en {len(result['predicciones'])} dias")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 6: Health check
    total += 1
    print("\nTest 6: Testing de salud del modelo...")
    try:
        result = test_model_health('900123456', df, 'Arroz 50kg')
        assert result['ok']
        print(f"  Salud: {result['salud']}")
        print(f"  MAE original: {result['mae_original']:.1f}, MAE actual: {result['mae_actual']:.1f}")
        print(f"  Degradacion: {result['degradacion_pct']:.1f}%")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 7: Reentrenamiento
    total += 1
    print("\nTest 7: Reentrenamiento con datos actualizados...")
    try:
        result = retrain_with_new_data('900123456', df, 'Arroz 50kg')
        assert result['ok']
        m = result['metricas']
        print(f"  MAE={m['MAE']:.1f}, MAPE={m['MAPE']:.1f}%")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 8: Multi-producto
    total += 1
    print("\nTest 8: Entrenar modelo para segundo producto...")
    try:
        result = train_model('900123456', df, 'Aceite caja x12')
        assert result['ok']
        print(f"  MAE={result['metricas']['MAE']:.1f}")
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Resumen
    print("\n" + "=" * 55)
    print(f"  RESULTADO: {passed}/{total} tests pasaron")
    if passed == total:
        print("  ✅ TODOS LOS TESTS PASARON")
    else:
        print(f"  ⚠️ {total - passed} tests fallaron")
    print("=" * 55)

if __name__ == '__main__':
    test_all()
