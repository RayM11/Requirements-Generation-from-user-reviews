import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns

# Resultados de 13 modelos evaluados en 4 (3 por ahora) datasets para la métrica F1-Score
# Las filas son datasets, y las columnas son modelos
f1_results = np.array([
    [0.884590752, 0.883297085, 0.83597018, 0.930634936, 0.932522285, 0.919247754, 0.923668953, 0.924037952, 0.929665945, 0.926956672, 0.925407177, 0.926820945, 0.91222117],
    [0.837843509, 0.892874898, 0.909300039, 0.944556143, 0.926956672, 0.93693594, 0.946118558, 0.939040527, 0.948111723, 0.951535013, 0.944309695, 0.915241869, 0.94913276],

    [0.82443696, 0.87920035, 0.882721225, 0.940325431, 0.946211598, 0.942654811, 0.921478836, 0.94304825, 0.951671236, 0.94800024, 0.949458112, 0.934837658, 0.945670663]
])

# Aplicar la prueba de Friedman
stat, p_value = friedmanchisquare(*f1_results.T)

print("Estadístico de Friedman:", stat)
print("p-valor:", p_value)

# Interpretación del p-valor
alpha = 0.05
if p_value < alpha:
    print("Hay diferencias significativas entre los modelos (rechazar H0).")
else:
    print("No hay diferencias significativas entre los modelos (no se rechaza H0).")

# Si hay diferencias significativas, realizar un análisis post-hoc (Test de Nemenyi)
if p_value < alpha:
    nemenyi_results = posthoc_nemenyi_friedman(f1_results)
    print("\nResultados del Test de Nemenyi (matriz de p-valores):\n")
    print(nemenyi_results)

# Graficas de comparación

# Diagrama de rangos promedio
# Calcular los rangos promedio de cada modelo
ranked_results = np.argsort(np.argsort(-f1_results, axis=1), axis=1) + 1  # Rangos (inverso para que el mejor sea 1)
mean_ranks = np.mean(ranked_results, axis=0)  # Promedio de rangos por modelo

# Etiquetas de modelos
model_labels = ["GP2-base", "ALBERT-large", "XLNET-base", "RoBERTa-base",
                "RoBERTa+L+RC", "RoBERTa+L+RP", "RoBERTa+MLP+RP",
                "RoBERTa+MLP+RC", "BERTweet-base", "BERTweet+L+RC",
                "BERTweet+L+RP", "BERTweet+MLP+RP", "BERTweet+MLP+RC"]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(model_labels, mean_ranks, color="skyblue")
plt.xlabel("Rango Promedio")
plt.title("Diagrama de Rangos Promedios por Modelo")
plt.gca().invert_yaxis()  # Invertir el eje y para mostrar el mejor modelo arriba
plt.show()


# Heatmap de Comparaciones Post-Hoc
if p_value < alpha:
    nemenyi_results = posthoc_nemenyi_friedman(f1_results)

    # Crear un heatmap para los p-valores
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_results, annot=True, cmap="coolwarm", fmt=".3f", xticklabels=model_labels, yticklabels=model_labels)
    plt.title("Heatmap de Comparaciones Post-Hoc (Test de Nemenyi)")
    plt.xlabel("Modelo")
    plt.ylabel("Modelo")
    plt.show()
