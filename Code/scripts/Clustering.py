import pandas as pd
import numpy as np
import umap
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from tqdm import tqdm
import os
import json
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# 1. Cargar los comentarios desde el CSV
def cargar_comentarios(ruta_csv):
    """Carga los comentarios desde un archivo CSV."""
    df = pd.read_csv(ruta_csv)
    # Asegurar que existe la columna "Review"
    if "Review" not in df.columns:
        raise ValueError("El CSV debe contener una columna llamada 'Review'")
    return df


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# 2. Generar embeddings utilizando un modelo pre-entrenado
def generar_embeddings(df, modelo="../models/BERTweet - base", batch_size=16):
    """Genera embeddings para los comentarios utilizando un modelo pre-entrenado."""
    # Cargar modelo y tokenizador
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModel.from_pretrained(modelo, trust_remote_code=True)

    # comments = df["Review"].tolist()

    # Mover modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Obtener embeddings por lotes para evitar problemas de memoria
    all_embeddings = []

    for i in tqdm(range(0, len(df), batch_size), desc="Generando embeddings"):
        batch_texts = df.iloc[i:i + batch_size]["Review"].tolist()

        # Tokenizar y obtener atención
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=200, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Obtener embeddings
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Usar la representación [CLS] como embedding del texto completo
        # embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    # Concatenar todos los embeddings
    embeddings_array = np.vstack(all_embeddings)

    # embeddings_array = model.encode(comments, prompt_name="passage")

    # Crear un DataFrame con los embeddings
    df_embeddings = pd.DataFrame({
        'id': range(len(df)),
        'embedding': [emb.tolist() for emb in embeddings_array]
    })

    return df_embeddings, embeddings_array


def generar_embeddings2(df, modelo="../models/BERTweet - base"):
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModel.from_pretrained(modelo, trust_remote_code=True)

    comments = df["Review"].tolist()

    # Tokenize sentences
    encoded_input = tokenizer(comments, padding=True, truncation=True, return_tensors='pt', max_length=200)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    df_embeddings = pd.DataFrame({
        'id': range(len(df)),
        'embedding': [emb.tolist() for emb in sentence_embeddings]
    })

    return df_embeddings, sentence_embeddings


def generar_embeddings_nomic(df, model_path="../models/nomic-embed-text", batch_size=16):
    """
    Genera embeddings utilizando el modelo nomic-embed-text-v1 a través de la interfaz de Hugging Face.

    Args:
        df: DataFrame con la columna "Review" que contiene los textos
        model_path: Nombre del modelo o ruta al modelo local
        batch_size: Tamaño del lote para procesamiento por batches

    Returns:
        df_embeddings: DataFrame con IDs y embeddings
        embeddings_array: Array NumPy con los embeddings
    """

    # Cargar el modelo y tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # Verificar disponibilidad de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Función para calcular embeddings con normalización
    def get_embeddings(batch_texts):
        # Tokenizar textos
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        # Calcular embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Usar la representación [CLS] (primera token) como embedding del texto
        embeddings = model_output.last_hidden_state[:, 0, :]

        # Normalizar embeddings (L2)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    # Preparar los textos
    textos = df["Review"].fillna("").tolist()
    total_batches = (len(textos) + batch_size - 1) // batch_size

    # Generar embeddings por lotes
    all_embeddings = []

    for i in tqdm(range(0, len(textos), batch_size), desc="Generando embeddings", total=total_batches):
        batch_texts = textos[i:i + batch_size]
        batch_embeddings = get_embeddings(batch_texts)
        all_embeddings.append(batch_embeddings)

    # Concatenar todos los embeddings
    embeddings_array = np.vstack(all_embeddings)

    # Crear DataFrame con IDs y embeddings
    df_embeddings = pd.DataFrame({
        'id': range(len(df)),
        'embedding': [emb.tolist() for emb in embeddings_array]
    })

    print(f"Generados embeddings de dimensión: {embeddings_array.shape}")

    return df_embeddings, embeddings_array


def reducir_dimensiones(embeddings_array, n_componentes=50, metodo='pca'):
    """
    Reduce la dimensionalidad de los embeddings.

    Args:
        embeddings_array: Array de embeddings original
        n_componentes: Número de componentes/dimensiones a mantener
        metodo: 'pca', 'tsne', o 'umap' para elegir el método de reducción

    Returns:
        embeddings_reducidos: Array con dimensionalidad reducida
    """

    print(f"Reduciendo dimensionalidad con {metodo} a {n_componentes} componentes...")

    if metodo == 'pca':
        # PCA es rápido y preserva la varianza global
        modelo = PCA(n_components=n_componentes, random_state=42)
        embeddings_reducidos = modelo.fit_transform(embeddings_array)
        varianza_explicada = sum(modelo.explained_variance_ratio_) * 100
        print(f"Varianza explicada por los {n_componentes} componentes: {varianza_explicada:.2f}%")

    elif metodo == 'tsne':
        # t-SNE preserva mejor la estructura local pero es más lento
        modelo = TSNE(n_components=n_componentes, random_state=42, n_jobs=-1)
        embeddings_reducidos = modelo.fit_transform(embeddings_array)

    return embeddings_reducidos


def determinar_k_optimo(embeddings_array, k_min=2, k_max=15, metodo='kmeans'):
    """
    Determina el número óptimo de clusters (k) usando el coeficiente de silueta.

    Args:
        embeddings_array: Array con los embeddings
        k_min: Mínimo número de clusters a probar
        k_max: Máximo número de clusters a probar
        metodo: 'kmeans' o 'fuzzy' para elegir el algoritmo de clustering

    Returns:
        k_optimo: Número óptimo de clusters
        resultados: Diccionario con los coeficientes de silueta para cada k
    """
    print("Determinando el número óptimo de clusters...")
    resultados = {}

    # Iterar sobre posibles valores de k
    for k in tqdm(range(k_min, k_max + 1), desc="Evaluando diferentes valores de k"):
        try:
            if metodo == 'kmeans':
                # Aplicar K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_array)

                # Calcular coeficiente de silueta
                score = silhouette_score(embeddings_array, labels)
                resultados[k] = score

            elif metodo == 'fuzzy':
                # Aplicar fuzzy c-means
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    embeddings_array.T, k, m=2, error=0.005, maxiter=1000, init=None
                )

                # Convertir a etiquetas duras para calcular el coeficiente de silueta
                labels = np.argmax(u, axis=0)

                # Calcular coeficiente de silueta
                score = silhouette_score(embeddings_array, labels)
                resultados[k] = score
        except Exception as e:
            print(f"Error al evaluar k={k}: {str(e)}")
            resultados[k] = -1  # Valor negativo para indicar error

    # Encontrar k con el mayor coeficiente de silueta
    k_optimo = max(resultados.items(), key=lambda x: x[1])[0]

    # Visualizar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(list(resultados.keys()), list(resultados.values()), 'o-')
    plt.axvline(x=k_optimo, color='r', linestyle='--', label=f'Óptimo: k={k_optimo}')
    plt.grid(True)
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Coeficiente de silueta')
    plt.title('Determinación del número óptimo de clusters')
    plt.legend()

    # Guardar gráfico
    plt.savefig('silhouette_analysis.png')
    plt.close()

    print(f"Número óptimo de clusters (k): {k_optimo}")
    return k_optimo, resultados


# 3. Aplicar algoritmo de agrupamiento fuzzy c-means
def agrupar_fuzzy_cmeans(embeddings_array, n_clusters=5, m=1.5, error=0.001, maxiter=2000):
    """
    Aplica el algoritmo fuzzy c-means a los embeddings con parámetros ajustados.

    Args:
        m: Coeficiente de difusión (1.0 = k-means, valores mayores = más difuso)
           Valores recomendados: 1.1-2.0
        error: Criterio de parada (menor valor = más iteraciones)
        maxiter: Número máximo de iteraciones
    """
    print(f"Aplicando fuzzy c-means con m={m}, error={error}, maxiter={maxiter}")

    # Aplicar fuzzy c-means
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        embeddings_array.T, n_clusters, m, error, maxiter, init=None
    )

    # Obtener el cluster con mayor grado de pertenencia para cada comentario
    cluster_memberships = np.argmax(u, axis=0)

    # Obtener los grados de pertenencia a cada cluster
    membership_values = u.T

    return cluster_memberships, membership_values, cntr


# Alternativa con K-means (más rápido y simple)
def agrupar_kmeans(embeddings_array, n_clusters=5):
    """Aplica el algoritmo K-means a los embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    return cluster_labels, kmeans.cluster_centers_


def agrupar_dbscan(embeddings_array, eps=0.5, min_samples=5):
    """
    Aplica DBSCAN, un algoritmo de clustering basado en densidad.
    Ventaja: No requiere especificar el número de clusters.
    """
    # Aplicar DBSCAN
    clustering = HDBSCAN(min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(embeddings_array)

    # Recalcular etiquetas para que empiecen desde 0 (DBSCAN usa -1 para ruido)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    n_clusters = len(unique_labels)
    print(f"DBSCAN encontró {n_clusters} clusters y {np.sum(labels == -1)} puntos de ruido")

    return labels, n_clusters


def agrupar_por_temas_lda(df, n_temas=10, max_features=10000):
    """
    Agrupa comentarios por temas utilizando LDA.
    Este enfoque es específico para texto y puede funcionar mejor que
    el clustering basado en embeddings para ciertos casos.
    """
    # Preprocesar textos
    textos = df['Review'].fillna('').tolist()

    # Crear matriz de términos-documentos
    vectorizador = CountVectorizer(max_features=max_features, stop_words='english')
    X = vectorizador.fit_transform(textos)

    # Aplicar LDA
    lda = LatentDirichletAllocation(n_components=n_temas, random_state=42)
    tema_distribucion = lda.fit_transform(X)

    # Asignar cada documento al tema con mayor probabilidad
    temas_asignados = np.argmax(tema_distribucion, axis=1)

    # Extraer palabras clave por tema
    feature_names = vectorizador.get_feature_names_out()
    palabras_por_tema = {}

    for idx_tema, tema in enumerate(lda.components_):
        top_palabras_idx = tema.argsort()[:-11:-1]  # Top 10 palabras
        top_palabras = [feature_names[i] for i in top_palabras_idx]
        palabras_por_tema[idx_tema] = top_palabras
        print(f"Tema {idx_tema}: {', '.join(top_palabras)}")

    return temas_asignados, tema_distribucion, palabras_por_tema


# 4. Crear el CSV final con comentarios, embeddings y clusters
def crear_csv_final(df_original, embeddings_array, cluster_labels, membership_values=None):
    """Crea un CSV final que contiene comentarios, embeddings y clusters."""
    df_final = df_original.copy()
    df_final['id'] = range(len(df_final))
    df_final['embedding'] = [emb.tolist() for emb in embeddings_array]
    df_final['cluster'] = cluster_labels

    # Si tenemos valores de pertenencia (fuzzy c-means)
    if membership_values is not None:
        for i in range(membership_values.shape[1]):
            df_final[f'membership_cluster_{i}'] = membership_values[:, i]

    return df_final


# 5. Función para obtener comentarios de un cluster específico
def obtener_comentarios_cluster(df_final, cluster_id):
    """Devuelve todos los comentarios pertenecientes a un cluster específico."""
    return df_final[df_final['cluster'] == cluster_id]


# 6. Función para analizar los clusters y extraer palabras clave
def analizar_clusters(df_final, num_palabras=10):
    """Analiza los clusters y extrae palabras clave para cada uno."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re

    # Preprocesar texto: eliminar signos de puntuación y convertir a minúsculas
    def preprocesar_texto(texto):
        if isinstance(texto, str):
            texto = re.sub(r'[^\w\s]', '', texto.lower())
            return texto
        return ""

    df_final['Review_procesado'] = df_final['Review'].apply(preprocesar_texto)

    resultados = {}
    clusters_unicos = df_final['cluster'].unique()

    for cluster in clusters_unicos:
        textos_cluster = df_final[df_final['cluster'] == cluster]['Review_procesado'].tolist()

        # Aplicar TF-IDF para extraer palabras clave
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(textos_cluster)
            feature_names = vectorizer.get_feature_names_out()

            # Calcular el promedio de TF-IDF para cada palabra en el cluster
            promedio_tfidf = tfidf_matrix.mean(axis=0).A1

            # Obtener las palabras con mayor puntuación TF-IDF
            indices_top = promedio_tfidf.argsort()[-num_palabras:][::-1]
            palabras_clave = [feature_names[i] for i in indices_top]

            # Guardar resultados
            resultados[int(cluster)] = {
                'palabras_clave': palabras_clave,
                'num_comentarios': len(textos_cluster),
                'ejemplos': df_final[df_final['cluster'] == cluster]['Review'].head(3).tolist()
            }
        except:
            resultados[int(cluster)] = {
                'palabras_clave': ["No se pudieron extraer palabras clave"],
                'num_comentarios': len(textos_cluster),
                'ejemplos': df_final[df_final['cluster'] == cluster]['Review'].head(3).tolist()
            }

    return resultados


def visualizar_clusters_2d(embeddings, clusters, nombre_archivo):
    """Visualiza clusters en 2D usando UMAP o t-SNE."""

    # Si los embeddings tienen más de 2 dimensiones, reducirlos a 2D para visualización
    if embeddings.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    # Configurar colores
    clusters_unicos = np.unique(clusters)
    n_clusters = len(clusters_unicos)

    # Manejar el caso de outliers (-1 en DBSCAN)
    if -1 in clusters_unicos:
        cmap = plt.cm.get_cmap('tab10', n_clusters - 1)
        colors = ['black'] + [cmap(i) for i in range(n_clusters - 1)]
    else:
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]

    # Crear la figura
    plt.figure(figsize=(12, 10))

    # Dibujar puntos para cada cluster
    for i, cluster_id in enumerate(clusters_unicos):
        mask = clusters == cluster_id
        color = colors[i]
        label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Outliers'

        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label,
            s=30,
            alpha=0.7
        )

    plt.title(f'Visualización de {n_clusters} clusters', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()


# Función principal que ejecuta todo el proceso
def procesar_comentarios(ruta_csv_entrada, ruta_csv_embeddings, ruta_csv_final,
                         modelo_embedding="../models/BERTweet - base",
                         # ruta_modelo_nomic="../models/nomic-embed-text",
                         n_clusters=None, method="fuzzy", ruta_analisis=None,
                         auto_k=True, k_min=2, k_max=15,
                         reducir_dim=True, n_componentes=50, metodo_reduccion='umap'
                         ):
    """Ejecuta el proceso completo de procesamiento de comentarios."""
    print("1. Cargando comentarios...")
    df = cargar_comentarios(ruta_csv_entrada)

    print("2. Generando embeddings...")
    # if modelo_embedding == "nomic":
    #     # Usar el modelo Nomic para embeddings
    #     df_embeddings, embeddings_array = generar_embeddings_nomic(
    #         df, model_path=ruta_modelo_nomic
    #     )
    # else:
    #     # Usar el enfoque original con transformers
    #     df_embeddings, embeddings_array = generar_embeddings(
    #         df,
    #         nombre_modelo=modelo_embedding
    #     )

    df_embeddings, embeddings_array = generar_embeddings2(df, modelo=modelo_embedding)

    # Reducir dimensionalidad si se solicita
    if reducir_dim:
        dimension_original = embeddings_array.shape[1]
        embeddings_array = reducir_dimensiones(
            embeddings_array,
            n_componentes=n_componentes,
            metodo=metodo_reduccion
        )
        print(f"Dimensionalidad reducida de {dimension_original} a {embeddings_array.shape[1]}")

    # Guardar embeddings en CSV
    df_embeddings_to_save = df_embeddings.copy()
    if reducir_dim:
        df_embeddings_to_save['embedding_reducido'] = [emb.tolist() for emb in embeddings_array]
    df_embeddings_to_save.to_csv(ruta_csv_embeddings, index=False)
    print(f"Embeddings guardados en {ruta_csv_embeddings}")

    # Determinar automáticamente el número de clusters si se solicita

    if method == "fuzzy" or method == "kmeans":
        if auto_k:
            k_optimo, resultados_k = determinar_k_optimo(
                embeddings_array, k_min=k_min, k_max=k_max, metodo=method
            )
            n_clusters = k_optimo

            # Guardar resultados de análisis de k
            with open('analisis_k_optimo.json', 'w') as f:
                json.dump({str(k): float(score) for k, score in resultados_k.items()}, f, indent=4)

        # Verificar que n_clusters tenga un valor
        if n_clusters is None:
            n_clusters = 5  # Valor por defecto si no se especificó ni se calculó automáticamente

        print(f"3. Aplicando algoritmo de agrupamiento con {n_clusters} clusters...")
        if method == "fuzzy":
            # Aplicar fuzzy c-means
            cluster_labels, membership_values, _ = agrupar_fuzzy_cmeans(embeddings_array, n_clusters)
            df_final = crear_csv_final(df, embeddings_array, cluster_labels, membership_values)
        else:
            # Aplicar K-means
            cluster_labels, _ = agrupar_kmeans(embeddings_array, n_clusters)
            df_final = crear_csv_final(df, embeddings_array, cluster_labels)

    elif method == "dbscan":
        cluster_labels, _ = agrupar_dbscan(embeddings_array, eps=0.5, min_samples=5)
        df_final = crear_csv_final(df, embeddings_array, cluster_labels)

    elif method == "lda":
        cluster_labels = agrupar_por_temas_lda(df, n_temas=10)
        df_final = crear_csv_final(df, embeddings_array, cluster_labels)

        # Visualizar clusters
    print("Generando visualización de clusters...")
    visualizar_clusters_2d(embeddings_array, cluster_labels, "visualizacion_clusters.png")

    # Guardar CSV final
    df_final.to_csv(ruta_csv_final, index=False)
    print(f"CSV final guardado en {ruta_csv_final}")

    # Analizar clusters y guardar resultados
    if ruta_analisis:
        print("4. Analizando clusters...")
        resultados_analisis = analizar_clusters(df_final)

        with open(ruta_analisis, 'w', encoding='utf-8') as f:
            json.dump(resultados_analisis, f, ensure_ascii=False, indent=4)

        print(f"Análisis de clusters guardado en {ruta_analisis}")

    return df_final


# Ejemplo de uso
if __name__ == "__main__":
    # Definir rutas de archivos
    ruta_csv_entrada = "../data/swiftkey_informative.csv"  # Reemplazar con la ruta real del CSV de entrada
    ruta_csv_embeddings = "../data/swiftkey_informative_embeddings.csv"
    ruta_csv_final = "../data/swiftkey_informative_analysis.csv"
    ruta_analisis = "clustering_analysis.json"
    ruta_modelo = "../models/BERTweet - base"
    ruta_modelo_nomic = "../models/nomic-embed-text"

    # Ejecutar el proceso completo
    # Ejecutar el proceso completo con Nomic
    df_final = procesar_comentarios(
        ruta_csv_entrada=ruta_csv_entrada,
        ruta_csv_embeddings=ruta_csv_embeddings,
        ruta_csv_final=ruta_csv_final,
        modelo_embedding=ruta_modelo_nomic,  # Usar modelo Nomic
        # ruta_modelo_nomic=ruta_modelo_nomic,  # Ruta o nombre del modelo
        n_clusters=None,
        method="fuzzy",
        ruta_analisis=ruta_analisis,
        auto_k=True,
        k_min=10,
        k_max=50,
        reducir_dim=True,
        n_componentes=2,
        metodo_reduccion='pca'
    )

    # Ejemplo: obtener comentarios del cluster 0
    comentarios_cluster_0 = obtener_comentarios_cluster(df_final, 0)
    print(f"Número de comentarios en el cluster 0: {len(comentarios_cluster_0)}")
    print("Primeros 3 comentarios del cluster 0:")
    print(comentarios_cluster_0['Review'].head(3))
