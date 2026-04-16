import pandas as pd
import numpy as np

"""
========================================
DATASET: CREDIT RISK SIMULATION
========================================

Descripción:
Este dataset simula información de clientes para un modelo de riesgo crediticio.
El objetivo es predecir si un cliente entrará en incumplimiento (default).

----------------------------------------
VARIABLES (FEATURES)
----------------------------------------

customer_age:
    Edad del cliente en años.
    Rango: 18 - 70

monthly_income:
    Ingreso mensual del cliente.
    Representa la capacidad de pago.
    Rango aproximado: 800 - 8000

total_debt:
    Deuda total actual del cliente.
    Incluye créditos, tarjetas, etc.
    Rango: 0 - 5000

active_loans_count:
    Número de créditos activos.
    Indica nivel de exposición financiera.
    Rango: 0 - 10

delinquency_count:
    Número de eventos de mora en el pasado.
    Variable altamente predictiva de riesgo.
    Rango: 0 - 5

employment_type:
    Tipo de empleo del cliente:
        - formal
        - informal
        - self_employed

----------------------------------------
VARIABLE OBJETIVO (TARGET)
----------------------------------------

default_flag:
    Indicador de incumplimiento:
        1 = cliente entra en default (alto riesgo)
        0 = cliente paga correctamente

    Lógica simulada:
        - Bajos ingresos + alta deuda → mayor riesgo
        - Alto historial de mora → mayor riesgo

----------------------------------------
NOTAS
----------------------------------------

- Dataset completamente simulado
- Inspirado en casos reales de riesgo crediticio
- Puede ser utilizado para entrenamiento de modelos de clasificación

========================================
"""

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "customer_age": np.random.randint(18, 70, n),
    "monthly_income": np.random.randint(800, 8000, n),
    "total_debt": np.random.randint(0, 5000, n),
    "active_loans_count": np.random.randint(0, 10, n),
    "delinquency_count": np.random.randint(0, 5, n),
    "employment_type": np.random.choice(
        ["formal", "informal", "self_employed"], n
    )
})

# variable objetivo
data["default_flag"] = (
    (
        (data["monthly_income"] < 2000) &
        (data["total_debt"] > 3000)
    ) |
    (data["delinquency_count"] >= 3)
).astype(int)

# guardar dataset
data.to_csv("data.csv", index=False)