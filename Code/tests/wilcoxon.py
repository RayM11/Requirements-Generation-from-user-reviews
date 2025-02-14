import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Resultados de 13 modelos evaluados en 3 datasets para la métrica F1-Score
f1_results = np.array([
    [0.884590752, 0.883297085, 0.83597018, 0.930634936, 0.932522285, 0.919247754, 0.923668953, 0.924037952, 0.929665945, 0.926956672, 0.925407177, 0.926820945, 0.91222117],
    [0.837843509, 0.892874898, 0.909300039, 0.944556143, 0.926956672, 0.93693594, 0.946118558, 0.939040527, 0.948111723, 0.951535013, 0.944309695, 0.915241869, 0.94913276],
    [0.82443696, 0.87920035, 0.882721225, 0.940325431, 0.946211598, 0.942654811, 0.921478836, 0.94304825, 0.951671236, 0.94800024, 0.949458112, 0.934837658, 0.945670663]
])

# Etiquetas de modelos
model_labels = ["GP2-base", "ALBERT-large", "XLNET-base", "RoBERTa-base",
                "RoBERTa+L+RC", "RoBERTa+L+RP", "RoBERTa+MLP+RP",
                "RoBERTa+MLP+RC", "BERTweet-base", "BERTweet+L+RC",
                "BERTweet+L+RP", "BERTweet+MLP+RP", "BERTweet+MLP+RC"]

# Crear combinaciones de modelos para comparar con Wilcoxon
model_combinations = list(itertools.combinations(range(len(model_labels)), 2))

# Almacenar resultados
wilcoxon_results = np.ones((len(model_labels), len(model_labels)))  # Matriz de p-valores

# Aplicar Wilcoxon a cada par de modelos
alpha = 0.05  # Nivel de significancia
for i in range(len(model_labels)):
    for j in range(i + 1, len(model_labels)):
        # Wilcoxon requiere que compares pares de datos
        stat, p_value = wilcoxon(f1_results[:, i], f1_results[:, j])
        wilcoxon_results[i, j] = p_value
        wilcoxon_results[j, i] = p_value  # La matriz es simétrica

# Convertir la matriz de p-valores en un DataFrame para visualizar
df_wilcoxon = pd.DataFrame(wilcoxon_results, index=model_labels, columns=model_labels)

# --- Gráfico 1: Heatmap de los p-valores del Test de Wilcoxon ---
plt.figure(figsize=(12, 10))
sns.heatmap(df_wilcoxon, annot=True, cmap="coolwarm", fmt=".3f", xticklabels=model_labels, yticklabels=model_labels)
plt.title("Mapa de Calor de Comparaciones de Modelos (Test de Wilcoxon)")
plt.xlabel("Modelo")
plt.ylabel("Modelo")
plt.show()

# --- Gráfico 2: Boxplot de la Distribución de F1-Score por Modelo ---
plt.figure(figsize=(14, 6))
df_f1 = pd.DataFrame(f1_results, columns=model_labels)  # Convertir resultados a DataFrame
df_f1.boxplot(rot=90, grid=False)  # Boxplot de los modelos
plt.title("Distribución de F1-Score por Modelo")
plt.ylabel("F1-Score")
plt.xlabel("Modelo")
plt.show()
