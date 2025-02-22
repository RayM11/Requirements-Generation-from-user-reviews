from scipy.stats import kruskal

# Supongamos que tienes estas 3 listas con los valores de f1-score
f1_dataset1 = [0.884590752, 0.883297085, 0.83597018, 0.930634936, 0.932522285, 0.919247754, 0.923668953, 0.924037952, 0.929665945, 0.926956672, 0.925407177, 0.926820945, 0.91222117]  # Valores de f1-score para dataset 1
f1_dataset2 = [0.837843509, 0.892874898, 0.909300039, 0.944556143, 0.926956672, 0.93693594, 0.946118558, 0.939040527, 0.948111723, 0.951535013, 0.944309695, 0.915241869, 0.94913276]  # Valores de f1-score para dataset 2
f1_dataset3 = [0.82443696, 0.87920035, 0.882721225, 0.940325431, 0.946211598, 0.942654811, 0.921478836, 0.94304825, 0.951671236, 0.94800024, 0.949458112, 0.934837658, 0.945670663]  # Valores de f1-score para dataset 3

# Reorganizar los datos por modelo
f1_by_model = []
for i in range(13):  # 13 modelos
    f1_model_i = [f1_dataset1[i], f1_dataset2[i], f1_dataset3[i]]  # f1-score del modelo i en los 3 datasets
    f1_by_model.append(f1_model_i)

# Aplicar la prueba de Kruskal-Wallis
statistic, p_value = kruskal(*f1_by_model)

# Mostrar resultados
print("Resultados de la prueba de Kruskal-Wallis:")
print(f"Estadístico H: {statistic:.4f}")
print(f"p-valor: {p_value:.4f}")

# Interpretación
if p_value < 0.05:
    print("Conclusión: Existen diferencias significativas entre los modelos (rechazamos H0).")
else:
    print("Conclusión: No hay diferencias significativas entre los modelos (no rechazamos H0).")