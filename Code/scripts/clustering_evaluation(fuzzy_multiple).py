import pandas as pd
import numpy as np
import os
import time
from itertools import product
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from skfuzzy import cmeans
import ast
import warnings

warnings.filterwarnings('ignore')


def evaluate_fuzzy_cmeans_clustering(csv_path, output_folder, k_min, k_max, membership_threshold=0.3):
    """
    Función de evaluación para Fuzzy C-Means clustering con exploración de parámetros.

    Parámetros:
    -----------
    csv_path : str
        Ruta al archivo CSV con columnas 'Review', 'Relevant' y 'embeddings'
    output_folder : str
        Carpeta donde se guardarán los resultados
    k_min : int
        Número mínimo de clusters a evaluar
    k_max : int
        Número máximo de clusters a evaluar
    membership_threshold : float
        Umbral de pertenencia para asignación múltiple (default: 0.3)

    Retorna:
    --------
    pd.DataFrame : DataFrame con los resultados de todas las combinaciones
    """

    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar datos
    print("Cargando datos...")
    df = pd.read_csv(csv_path)

    # Eliminar textos duplicados
    print("Eliminando textos duplicados...")
    original_size = len(df)
    df = df.drop_duplicates(subset=['Review'], keep='first').reset_index(drop=True)
    duplicates_removed = original_size - len(df)
    print(f"Dataset original: {original_size} registros")
    print(f"Duplicados eliminados: {duplicates_removed}")
    print(f"Dataset final: {len(df)} registros")

    # Procesar embeddings (asumiendo que están como strings de listas)
    print("Procesando embeddings...")
    if isinstance(df['embeddings'].iloc[0], str):
        embeddings = np.array([ast.literal_eval(emb) for emb in df['embeddings']])
    else:
        embeddings = np.array(df['embeddings'].tolist())

    # Definir configuraciones de reducción de dimensionalidad
    reduction_configs = {
        'none': {'method': None, 'components': [None]},
        'pca': {'method': 'pca', 'components': [2, 10, 50, 100]},
        'umap': {'method': 'umap', 'components': [2, 10, 50, 100]}
    }

    # Lista para almacenar resultados
    results = []

    # Iterar sobre todas las combinaciones
    total_combinations = sum(len(config['components']) for config in reduction_configs.values()) * (k_max - k_min + 1)
    current_combination = 0

    for reduction_name, config in reduction_configs.items():
        for n_components in config['components']:
            for k in range(k_min, k_max + 1):
                current_combination += 1
                print(f"Procesando combinación {current_combination}/{total_combinations}: "
                      f"{reduction_name}_comp{n_components}_k{k}")

                start_time = time.time()

                try:
                    # Aplicar reducción de dimensionalidad
                    if config['method'] is None:
                        # Sin reducción
                        X_reduced = embeddings
                        reduction_str = "none"
                        comp_str = "None"
                    elif config['method'] == 'pca':
                        # PCA
                        if n_components >= embeddings.shape[1]:
                            print(
                                f"Saltando PCA con {n_components} componentes (mayor que dimensión original {embeddings.shape[1]})")
                            continue
                        pca = PCA(n_components=n_components, random_state=42)
                        X_reduced = pca.fit_transform(embeddings)
                        reduction_str = "pca"
                        comp_str = str(n_components)
                    elif config['method'] == 'umap':
                        # UMAP
                        if n_components >= embeddings.shape[0]:
                            print(f"Saltando UMAP con {n_components} componentes (mayor que número de muestras)")
                            continue
                        umap_reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15)
                        X_reduced = umap_reducer.fit_transform(embeddings)
                        reduction_str = "umap"
                        comp_str = str(n_components)

                    # Aplicar Fuzzy C-Means
                    cntr, u, u0, d, jm, p, fpc = cmeans(
                        X_reduced.T, k, 2, error=0.005, maxiter=1000, init=None, seed=42
                    )

                    # Asignar clusters con umbral múltiple
                    cluster_assignments = assign_clusters_with_threshold(u, membership_threshold)

                    # Crear dataset expandido para evaluación
                    expanded_data, expanded_labels = create_expanded_dataset(
                        X_reduced, cluster_assignments, df
                    )

                    # Calcular métricas de evaluación
                    if len(np.unique(expanded_labels)) > 1 and len(expanded_labels) > 1:
                        silhouette = silhouette_score(expanded_data, expanded_labels)
                        calinski_harabasz = calinski_harabasz_score(expanded_data, expanded_labels)
                        davies_bouldin = davies_bouldin_score(expanded_data, expanded_labels)
                        n_clusters_found = len(np.unique(expanded_labels))
                    else:
                        silhouette = -1.0
                        calinski_harabasz = 0.0
                        davies_bouldin = float('inf')
                        n_clusters_found = 1

                    execution_time = time.time() - start_time

                    # Crear ID de combinación
                    combination_id = f"fuzzy_cmeans_{reduction_str}_comp{comp_str}_k{k}"

                    # Guardar resultado
                    result = {
                        'combination_id': combination_id,
                        'algorithm': 'fuzzy_cmeans',
                        'reduction_method': reduction_str,
                        'n_components': float(n_components) if n_components is not None else None,
                        'k_value': float(k),
                        'n_clusters_found': n_clusters_found,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin,
                        'execution_time': execution_time
                    }

                    results.append(result)

                    # Guardar asignaciones detalladas
                    save_detailed_results(
                        df, cluster_assignments, u, output_folder, combination_id
                    )

                    print(f"✓ Completado: Silhouette={silhouette:.4f}, "
                          f"CH={calinski_harabasz:.2f}, DB={davies_bouldin:.4f}")

                except Exception as e:
                    print(f"✗ Error en combinación {combination_id}: {str(e)}")
                    continue

    # Guardar información sobre duplicados eliminados
    duplicates_info = {
        'original_dataset_size': original_size,
        'duplicates_removed': duplicates_removed,
        'final_dataset_size': len(df),
        'duplicate_removal_percentage': (duplicates_removed / original_size) * 100 if original_size > 0 else 0
    }

    # Guardar información de duplicados
    duplicates_info_path = os.path.join(output_folder, 'duplicates_removal_info.txt')
    with open(duplicates_info_path, 'w', encoding='utf-8') as f:
        f.write("INFORMACIÓN SOBRE ELIMINACIÓN DE DUPLICADOS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Tamaño del dataset original: {duplicates_info['original_dataset_size']}\n")
        f.write(f"Duplicados eliminados: {duplicates_info['duplicates_removed']}\n")
        f.write(f"Tamaño del dataset final: {duplicates_info['final_dataset_size']}\n")
        f.write(f"Porcentaje de duplicados removidos: {duplicates_info['duplicate_removal_percentage']:.2f}%\n")

    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)

    # Guardar resultados principales
    summary_path = os.path.join(output_folder, 'clustering_results_summary_fuzzy_cmeans.csv')
    results_df.to_csv(summary_path, index=False)

    # Generar reporte de mejores resultados
    generate_best_results_report(results_df, output_folder)

    print(f"\n✓ Evaluación completada! Resultados guardados en: {output_folder}")
    print(f"Total de combinaciones evaluadas: {len(results_df)}")

    return results_df


def assign_clusters_with_threshold(membership_matrix, threshold):
    """
    Asigna clusters basado en umbral de pertenencia múltiple.

    Parámetros:
    -----------
    membership_matrix : np.array
        Matriz de pertenencias de Fuzzy C-Means (k x n)
    threshold : float
        Umbral de pertenencia

    Retorna:
    --------
    list : Lista de listas con asignaciones de cluster para cada punto
    """
    assignments = []

    for i in range(membership_matrix.shape[1]):  # Para cada punto de datos
        memberships = membership_matrix[:, i]

        # Encontrar clusters que superan el umbral
        valid_clusters = np.where(memberships >= threshold)[0]

        if len(valid_clusters) == 0:
            # Si ningún cluster supera el umbral, asignar al de mayor pertenencia
            assignments.append([np.argmax(memberships)])
        else:
            # Asignar a todos los clusters que superan el umbral
            assignments.append(valid_clusters.tolist())

    return assignments


def create_expanded_dataset(X, cluster_assignments, original_df):
    """
    Crea un dataset expandido duplicando puntos asignados a múltiples clusters.

    Parámetros:
    -----------
    X : np.array
        Datos reducidos
    cluster_assignments : list
        Asignaciones de cluster por punto
    original_df : pd.DataFrame
        DataFrame original

    Retorna:
    --------
    tuple : (datos_expandidos, etiquetas_expandidas)
    """
    expanded_data = []
    expanded_labels = []
    expanded_indices = []

    for i, clusters in enumerate(cluster_assignments):
        for cluster in clusters:
            expanded_data.append(X[i])
            expanded_labels.append(cluster)
            expanded_indices.append(i)

    return np.array(expanded_data), np.array(expanded_labels)


def save_detailed_results(df, cluster_assignments, membership_matrix, output_folder, combination_id):
    """
    Guarda resultados detallados de la asignación de clusters.
    """
    # Crear DataFrame con resultados detallados
    detailed_results = []

    for i, (_, row) in enumerate(df.iterrows()):
        clusters = cluster_assignments[i]
        memberships = membership_matrix[:, i]

        for cluster in clusters:
            detailed_results.append({
                'original_index': i,
                'review': row['Review'],
                # 'relevant': row['Relevant'],
                'assigned_cluster': cluster,
                'membership_score': memberships[cluster],
                'max_membership': np.max(memberships),
                'n_clusters_assigned': len(clusters)
            })

    detailed_df = pd.DataFrame(detailed_results)

    # Guardar archivo detallado
    detailed_path = os.path.join(output_folder, f'detailed_{combination_id}.csv')
    detailed_df.to_csv(detailed_path, index=False)


def generate_best_results_report(results_df, output_folder):
    """
    Genera un reporte con los mejores resultados según diferentes métricas.
    """
    report = []

    # Mejor Silhouette Score
    best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
    report.append("MEJORES RESULTADOS POR MÉTRICA")
    report.append("=" * 50)
    report.append(f"\nMejor Silhouette Score: {best_silhouette['silhouette_score']:.4f}")
    report.append(f"Combinación: {best_silhouette['combination_id']}")

    # Mejor Calinski-Harabasz Score
    best_ch = results_df.loc[results_df['calinski_harabasz_score'].idxmax()]
    report.append(f"\nMejor Calinski-Harabasz Score: {best_ch['calinski_harabasz_score']:.2f}")
    report.append(f"Combinación: {best_ch['combination_id']}")

    # Mejor Davies-Bouldin Score (menor es mejor)
    valid_db = results_df[results_df['davies_bouldin_score'] != float('inf')]
    if not valid_db.empty:
        best_db = valid_db.loc[valid_db['davies_bouldin_score'].idxmin()]
        report.append(f"\nMejor Davies-Bouldin Score: {best_db['davies_bouldin_score']:.4f}")
        report.append(f"Combinación: {best_db['combination_id']}")

    # Top 5 combinaciones por Silhouette Score
    report.append(f"\n\nTOP 5 COMBINACIONES (Silhouette Score)")
    report.append("-" * 50)
    top_5 = results_df.nlargest(5, 'silhouette_score')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        report.append(f"{i}. {row['combination_id']}: {row['silhouette_score']:.4f}")

    # Guardar reporte
    report_path = os.path.join(output_folder, 'best_results_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración de ejemplo
    csv_path = "../data/transfermovil_informative_with_embeddings (nomic).csv"  # Reemplazar con la ruta real
    output_folder = "../data/fuzzy_cmeans_multiple_results (transfermovil)"
    k_min = 2
    k_max = 10
    membership_threshold = 0.3  # Umbral de pertenencia para asignación múltiple

    # Ejecutar evaluación
    results = evaluate_fuzzy_cmeans_clustering(
        csv_path=csv_path,
        output_folder=output_folder,
        k_min=k_min,
        k_max=k_max,
        membership_threshold=membership_threshold
    )

    print("\nPrimeras 5 filas de resultados:")
    print(results.head())

    # input_csv_path = "../data/swiftkey_informative_with_embeddings (nomic).csv"
    # output_folder = "../data/fuzzy_cmeans_multiple_results (swiftkey)"
