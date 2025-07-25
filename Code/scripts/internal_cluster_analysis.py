import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
from datetime import datetime
import os


class ClusterTextAnalyzer:
    def __init__(self, csv_file_path, text_column, cluster_column):
        """
        Inicializa el analizador de clusters de texto

        Args:
            csv_file_path (str): Ruta al archivo CSV con columnas 'text' y 'cluster'
        """
        self.df = pd.read_csv(csv_file_path)
        self.df.columns = self.df.columns.str.strip()  # Limpiar nombres de columnas

        # Verificar que existen las columnas necesarias
        if text_column not in self.df.columns or cluster_column not in self.df.columns:
            raise ValueError(f"El CSV debe contener las columnas '{text_column}' y '{cluster_column}'")

        # Limpiar datos nulos
        self.df = self.df.dropna(subset=[text_column, cluster_column])
        self.clusters = sorted(self.df[cluster_column].unique())

    def preprocess_text(self, text):
        """
        Preprocesa el texto eliminando caracteres especiales y normalizando
        """
        if pd.isna(text):
            return ""

        # Convertir a minúsculas
        text = str(text).lower()

        # Mantener solo letras, números y espacios (ajustar según idioma)
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)

        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenizar el texto
        tokens = word_tokenize(text)

        # Eliminación de stopwords
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]

        # juntar los tokens de nuevo
        text = ' '.join(tokens)

        return text

    def extract_ngrams(self, texts, n=2, top_k=10):
        """
        Extrae los n-gramas más frecuentes de una lista de textos
        """
        # Unir todos los textos
        combined_text = ' '.join([self.preprocess_text(text) for text in texts])

        # Dividir en palabras
        words = combined_text.split()

        # Generar n-gramas
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            if len(ngram.strip()) > 0:
                ngrams.append(ngram)

        # Contar frecuencias
        ngram_counts = Counter(ngrams)

        return ngram_counts.most_common(top_k)

    def analyze_cluster_tfidf(self, cluster_id, text_column, cluster_column, max_features=50):
        """
        Analiza un cluster específico usando TF-IDF
        """
        # Obtener textos del cluster
        cluster_texts = self.df[self.df[cluster_column] == cluster_id][text_column].tolist()

        if not cluster_texts:
            return None, None

        # Preprocesar textos
        processed_texts = [self.preprocess_text(text) for text in cluster_texts]

        # Aplicar TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,  # Ajustar según idioma
            ngram_range=(1, 1),
            min_df=1
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Calcular scores promedio de TF-IDF para cada término
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Crear diccionario de términos y scores
            tfidf_scores = dict(zip(feature_names, mean_scores))

            # Ordenar por score descendente
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

            return sorted_terms, len(cluster_texts)

        except Exception as e:
            print(f"Error en TF-IDF para cluster {cluster_id}: {e}")
            return None, len(cluster_texts)

    def analyze_all_clusters(self, text_column, cluster_column, output_file='cluster_analysis_results.txt', top_terms=20, top_ngrams=10):
        """
        Analiza todos los clusters y guarda los resultados
        """
        results = []

        print("Iniciando análisis de clusters...")

        for cluster_id in self.clusters:
            print(f"Analizando cluster {cluster_id}...")

            # Análisis TF-IDF
            tfidf_results, cluster_size = self.analyze_cluster_tfidf(cluster_id, text_column, cluster_column, max_features=100)

            # Obtener textos del cluster para n-gramas
            cluster_texts = self.df[self.df[cluster_column] == cluster_id][text_column].tolist()

            # Análisis de bigramas
            bigrams = self.extract_ngrams(cluster_texts, n=2, top_k=top_ngrams)

            # Análisis de trigramas
            trigrams = self.extract_ngrams(cluster_texts, n=3, top_k=top_ngrams)

            # Estadísticas básicas
            avg_length = np.mean([len(str(text)) for text in cluster_texts])

            cluster_result = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'avg_text_length': avg_length,
                'tfidf_terms': tfidf_results[:top_terms] if tfidf_results else [],
                'top_bigrams': bigrams,
                'top_trigrams': trigrams,
                'sample_texts': cluster_texts[:3]  # Primeros 3 textos como muestra
            }

            results.append(cluster_result)

        # Guardar resultados
        self.save_results(results, output_file)

        return results

    def save_results(self, results, output_file):
        """
        Guarda los resultados del análisis en un archivo de texto
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ANÁLISIS DE CLUSTERS DE TEXTO\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Resumen general
            total_texts = sum([r['size'] for r in results])
            f.write(f"RESUMEN GENERAL:\n")
            f.write(f"- Total de clusters: {len(results)}\n")
            f.write(f"- Total de textos: {total_texts}\n")
            f.write(f"- Promedio de textos por cluster: {total_texts / len(results):.1f}\n\n")

            # Análisis por cluster
            for result in results:
                f.write("-" * 60 + "\n")
                f.write(f"CLUSTER {result['cluster_id']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Tamaño: {result['size']} textos\n")
                f.write(f"Longitud promedio: {result['avg_text_length']:.1f} caracteres\n\n")

                # Términos TF-IDF más relevantes
                f.write("TÉRMINOS MÁS RELEVANTES (TF-IDF):\n")
                if result['tfidf_terms']:
                    for i, (term, score) in enumerate(result['tfidf_terms'], 1):
                        f.write(f"{i:2d}. {term:<20} (score: {score:.4f})\n")
                else:
                    f.write("No se pudieron calcular términos TF-IDF\n")
                f.write("\n")

                # Bigramas más frecuentes
                f.write("BIGRAMAS MÁS FRECUENTES:\n")
                for i, (bigram, count) in enumerate(result['top_bigrams'], 1):
                    f.write(f"{i:2d}. '{bigram}' ({count} veces)\n")
                f.write("\n")

                # Trigramas más frecuentes
                f.write("TRIGRAMAS MÁS FRECUENTES:\n")
                for i, (trigram, count) in enumerate(result['top_trigrams'], 1):
                    f.write(f"{i:2d}. '{trigram}' ({count} veces)\n")
                f.write("\n")

                # Ejemplos de textos
                f.write("EJEMPLOS DE TEXTOS:\n")
                for i, text in enumerate(result['sample_texts'], 1):
                    f.write(f"{i}. {str(text)[:200]}{'...' if len(str(text)) > 200 else ''}\n")
                f.write("\n\n")

        print(f"Resultados guardados en: {output_file}")


# Función principal para ejecutar el análisis
def main():
    # Configuración
    csv_file = "../data/fuzzy_cmeans_multiple_results (transfermovil)/detailed_fuzzy_cmeans_pca_comp2_k3.csv"  # Cambiar por la ruta de tu archivo
    output_file = "transfermovil_cluster_analysis_results.txt"
    text_column = "review"
    cluster_column = "assigned_cluster"

    try:
        # Crear analizador
        analyzer = ClusterTextAnalyzer(csv_file, text_column, cluster_column)

        print(f"Datos cargados: {len(analyzer.df)} textos en {len(analyzer.clusters)} clusters")

        # Ejecutar análisis
        results = analyzer.analyze_all_clusters(
            text_column,
            cluster_column,
            output_file=output_file,
            top_terms=10,  # Top 10 términos TF-IDF por cluster
            top_ngrams=10  # Top 10 n-gramas por cluster
        )

        print(f"\nAnálisis completado. Revisa el archivo '{output_file}' para ver los resultados.")

        # Mostrar resumen en consola
        print("\nRESUMEN POR CLUSTER:")
        for result in results:
            print(f"Cluster {result['cluster_id']}: {result['size']} textos")
            if result['tfidf_terms']:
                top_3_terms = [term for term, score in result['tfidf_terms'][:3]]
                print(f"  Top términos: {', '.join(top_3_terms)}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_file}'")
        print("Asegúrate de que el archivo existe y tiene las columnas 'text' y 'cluster'")
    except Exception as e:
        print(f"Error durante el análisis: {e}")


if __name__ == "__main__":
    main()
