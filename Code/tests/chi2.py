import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Configuración inicial
n_models = 3  # Número de modelos (ej: BERT, RoBERTa, XLNet)
n_datasets = 4  # Número de datasets
total_samples = 3000  # Total de muestras por dataset

# --------------------------------------------------------------------
# PASO 1: Simular TUS VALORES DE ACCURACY (reemplázalos con tus datos reales)
# Ejemplo: accuracy de 3 modelos en 4 datasets (valores entre 0 y 1)
accuracy_data = {
    'Dataset_1: Facebook': {
        'Model_1': 0.9000,
        'Model_2': 0.8983,
        'Model_3': 0.8467,
        'Model_4': 0.9397,
        'Model_5': 0.941,
        'Model_6': 0.93,
        'Model_7': 0.931,
        'Model_8': 0.932333333,
        'Model_9': 0.9377,
        'Model_10': 0.934666667,
        'Model_11': 0.933,
        'Model_12': 0.934,
        'Model_13': 0.922666667
    },
    'Dataset_2: SwiftKey': {
        'Model_1': 0.7487,
        'Model_2': 0.8419,
        'Model_3': 0.8660,
        'Model_4': 0.9217,
        'Model_5': 0.910666667,
        'Model_6': 0.926666667,
        'Model_7': 0.923666667,
        'Model_8': 0.916666667,
        'Model_9': 0.9247,
        'Model_10': 0.931333333,
        'Model_11': 0.923333333,
        'Model_12': 0.878,
        'Model_13': 0.928
    },
    'Dataset_3: TempleRun2': {
        'Model_1': 0.8980,
        'Model_2': 0.8140,
        'Model_3': 0.8360,
        'Model_4': 0.9137,
        'Model_5': 0.924,
        'Model_6': 0.919,
        'Model_7': 0.888666667,
        'Model_8': 0.919,
        'Model_9': 0.9320,
        'Model_10': 0.925333333,
        'Model_11': 0.928333333,
        'Model_12': 0.911,
        'Model_13': 0.924
    }#,
#    'Dataset_4': {
#        'Model_1': 0.85,
#        'Model_2': 0.84,
#        'Model_3': 0.87,
#        'Model_4': 0.84,
#        'Model_5': 0.84,
#        'Model_6': 0.84,
#        'Model_7': 0.84,
#        'Model_8': 0.84,
#        'Model_9': 0.84,
#        'Model_10': 0.84,
#        'Model_11': 0.84,
#        'Model_12': 0.84,
#        'Model_13': 0.84
#    }
}


# --------------------------------------------------------------------
# PASO 2: Generar conteos de correctos/incorrectos desde el accuracy
def generate_counts(accuracy_dict, total_samples):
    counts_dict = {}
    for dataset, models in accuracy_dict.items():
        counts_dict[dataset] = {}
        for model, acc in models.items():
            correct = int(acc * total_samples)  # Convertir accuracy a conteo
            incorrect = total_samples - correct
            counts_dict[dataset][model] = {
                'correct': correct,
                'incorrect': incorrect
            }
    return counts_dict


# Generar los datos realistas
data = generate_counts(accuracy_data, total_samples)

# --------------------------------------------------------------------
# PASO 3: Realizar prueba de Chi-cuadrado para cada dataset
results = {}
for dataset in data:
    # Construir tabla de contingencia: modelos vs correctos/incorrectos
    contingency_table = []
    for model in data[dataset]:
        correct = data[dataset][model]['correct']
        incorrect = data[dataset][model]['incorrect']
        contingency_table.append([correct, incorrect])

    # Aplicar prueba de Chi-cuadrado
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    results[dataset] = {
        'Chi2 Statistic': chi2_stat,
        'p-value': p_value,
        'Significativo (α=0.05)': p_value < 0.05
    }

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results).T
print("Resultados de la prueba Chi-cuadrado por dataset:")
print(results_df)

# --------------------------------------------------------------------
# Ejemplo de salida (con los datos de ejemplo):
"""
Resultados de la prueba Chi-cuadrado por dataset:
               Chi2 Statistic   p-value  Significativo (α=0.05)
Dataset_1           12.345678  0.002134                     True
Dataset_2            5.432109  0.065432                    False
Dataset_3           18.765432  0.000015                     True
Dataset_4            3.210987  0.201345                    False
"""