# text_clustering.py
import warnings

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
from sklearn.metrics import silhouette_score
from Code.logic.commentClustering.embeddingsGeneration import generate_embeddings, load_embeddings_from_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringModule:
    """
    A class for clustering text reviews using embeddings and various clustering algorithms.
    Includes silhouette-based K selection for optimal cluster count.

    Attributes:
        available_reductions: List of supported dimensionality reduction methods
        available_clusterings: List of supported clustering algorithms
    """

    def __init__(self):
        self.available_reductions = ['None', 'PCA', 'UMAP']
        self.available_clusterings = ['k-means', 'fuzzy c-means', 'agglomerative']

    def cluster_reviews(
            self,
            df: pd.DataFrame,
            clustering_algorithm: str = 'k-means',
            dim_reduction: str = 'None',
            k_min: int = 3,
            k_max: int = 10,
            model_name: str = "nomic-embed-text",
            batch_size: int = 32,
            max_length: int = 512,
            reduction_kwargs: Optional[Dict] = None,
            clustering_kwargs: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Cluster reviews using optimal K selected by silhouette score.

        Args:
            df: DataFrame containing 'Review' column with text data
            clustering_algorithm: Clustering method to use
                ('k-means', 'fuzzy c-means', or 'agglomerative')
            dim_reduction: Dimensionality reduction method
                ('None', 'PCA', or 'UMAP')
            k_min: Minimum number of clusters to try
            k_max: Maximum number of clusters to try
            model_name: Name of the embedding model to use
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length for tokenization
            reduction_kwargs: Additional arguments for dimensionality reduction
            clustering_kwargs: Additional arguments for clustering algorithm

        Returns:
            pd.DataFrame: Original DataFrame with added 'cluster' column

        Raises:
            ValueError: For invalid parameters or empty data
        """
        # Validate input parameters
        if 'Review' not in df.columns:
            raise ValueError("DataFrame must contain 'Review' column")

        if dim_reduction not in self.available_reductions:
            raise ValueError(f"Invalid dim_reduction. Choose from: {self.available_reductions}")

        if clustering_algorithm not in self.available_clusterings:
            raise ValueError(f"Invalid clustering_algorithm. Choose from: {self.available_clusterings}")

        if k_min < 2 or k_max < k_min:
            raise ValueError("k_min must be >=2 and k_max must be >= k_min")

        # Generate embeddings
        logger.info("Generating text embeddings...")
        df_emb = generate_embeddings(
            df,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length
        )
        _, embeddings = load_embeddings_from_dataframe(df_emb)

        # Apply dimensionality reduction
        logger.info(f"Applying dimensionality reduction: {dim_reduction}")
        reduced_embeddings = self._reduce_dimensionality(
            embeddings,
            method=dim_reduction,
            n_components=2,
            kwargs=reduction_kwargs or {}
        )

        # Find optimal K using silhouette score
        logger.info(f"Finding optimal K between {k_min} and {k_max} using silhouette score")
        optimal_k, best_score = self._find_optimal_k(
            reduced_embeddings,
            clustering_algorithm,
            k_min,
            k_max,
            clustering_kwargs or {}
        )
        logger.info(f"Selected K={optimal_k} with silhouette score={best_score:.4f}")

        # Apply clustering with optimal K
        logger.info(f"Clustering with {clustering_algorithm} using K={optimal_k}...")
        cluster_labels = self._apply_clustering(
            reduced_embeddings,
            method=clustering_algorithm,
            n_clusters=optimal_k,
            kwargs=clustering_kwargs or {}
        )

        # Assign cluster labels to DataFrame
        result_df = df_emb.copy()
        if clustering_algorithm == 'fuzzy c-means':
            result_df = self._handle_fuzzy_labels(result_df, cluster_labels)
        else:
            result_df['Cluster'] = cluster_labels

        # Remove embeddings column
        result_df = result_df[['Review', 'Cluster']]
        result_df = result_df.drop_duplicates()
        # Sort elements by cluster and adding a ID
        result_df = result_df.sort_values(by="Cluster", ascending=True)
        result_df['ID'] = range(1, len(result_df) + 1)
        columns = ['ID'] + [col for col in result_df.columns if col != 'ID']
        result_df = result_df[columns]

        return result_df

    def _reduce_dimensionality(
            self,
            embeddings: np.ndarray,
            method: str,
            n_components: int = 2,
            kwargs: Optional[Dict] = None
    ) -> np.ndarray:
        """Reduce dimensionality of embeddings using specified method."""
        kwargs = kwargs or {}

        if method == 'None':
            return embeddings

        elif method == 'PCA':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, **kwargs)
            return reducer.fit_transform(embeddings)

        elif method == 'UMAP':
            try:
                import umap
            except ImportError:
                raise ImportError("UMAP not installed. Use 'pip install umap-learn'")

            reducer = umap.UMAP(n_components=n_components, **kwargs)
            return reducer.fit_transform(embeddings)

    def _apply_clustering(
            self,
            data: np.ndarray,
            method: str,
            n_clusters: int,
            kwargs: Optional[Dict] = None
    ) -> np.ndarray:
        """Apply clustering algorithm to data."""
        kwargs = kwargs or {}

        if method == 'k-means':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            return kmeans.fit_predict(data)

        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            agg = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            return agg.fit_predict(data)

        elif method == 'fuzzy c-means':
            try:
                import skfuzzy as fuzz
            except ImportError:
                raise ImportError("scikit-fuzzy not installed. Use 'pip install scikit-fuzzy'")

            # Transpose data for fuzzy c-means (features x samples)
            data_t = data.T
            m = kwargs.get('m', 2.0)  # Fuzziness parameter

            # Run fuzzy c-means
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_t, n_clusters, m, error=0.005, maxiter=1000, **kwargs
            )
            return u.T  # Return membership matrix (samples x clusters)

    def _handle_fuzzy_labels(
            self,
            df: pd.DataFrame,
            membership: np.ndarray,
            threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Handle fuzzy clustering results by creating duplicate rows for multi-cluster membership.

        Args:
            df: Original DataFrame
            membership: Membership matrix (n_samples x n_clusters)
            threshold: Membership threshold for assigning to cluster

        Returns:
            pd.DataFrame: Modified DataFrame with cluster assignments
        """
        records = []
        for i, row in df.iterrows():
            max_cluster = None
            max_value = 0

            for cluster_idx, value in enumerate(membership[i]):
                if value >= threshold:
                    # Create duplicate row for each qualified cluster
                    new_row = row.copy()
                    new_row['Cluster'] = cluster_idx
                    records.append(new_row)

                # Track cluster with highest membership
                if value > max_value:
                    max_value = value
                    max_cluster = cluster_idx

            # If no cluster meets threshold, assign to highest membership cluster
            if not records or records[-1]['Cluster'] != max_cluster:
                new_row = row.copy()
                new_row['Cluster'] = max_cluster
                records.append(new_row)

        return pd.DataFrame(records).reset_index(drop=True)

    def _find_optimal_k(
            self,
            data: np.ndarray,
            algorithm: str,
            k_min: int,
            k_max: int,
            clustering_kwargs: Dict
    ) -> Tuple[int, float]:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            data: Input data for clustering
            algorithm: Clustering algorithm to use
            k_min: Minimum number of clusters to try
            k_max: Maximum number of clusters to try
            clustering_kwargs: Arguments for clustering algorithm

        Returns:
            Tuple[int, float]: Optimal K and its silhouette score
        """
        best_k = k_min
        best_score = -1.0
        k_scores = []

        # Try each K value in range
        for k in range(k_min, k_max + 1):
            try:
                # Apply clustering
                labels = self._apply_clustering(
                    data,
                    method=algorithm,
                    n_clusters=k,
                    kwargs=clustering_kwargs
                )

                # For fuzzy clustering, convert to hard labels
                if algorithm == 'fuzzy c-means':
                    hard_labels = np.argmax(labels, axis=1)
                else:
                    hard_labels = labels

                # Calculate silhouette score
                score = silhouette_score(data, hard_labels)
                k_scores.append((k, score))
                logger.info(f"K={k} - Silhouette Score: {score:.4f}")

                # Update best K if improved
                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.warning(f"Failed clustering for K={k}: {str(e)}")
                continue

        # Log all scores for debugging
        logger.debug(f"All silhouette scores: {k_scores}")
        return best_k, best_score


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Create sample data
    sample_data = pd.DataFrame({
        'Review': [
            "This app is amazing! Great features and functionality.",
            "Terrible app, crashes all the time.",
            "Good user interface, easy to navigate.",
            "Not worth the download, very buggy.",
            "Excellent performance and stability."
        ]
    })

    # Initialize clusterer
    clusterMaker = ClusteringModule()

    # Cluster reviews with optimal K selection
    result_df = clusterMaker.cluster_reviews(
        df=sample_data,
        clustering_algorithm='fuzzy c-means',
        dim_reduction='UMAP',
        k_min=2,
        k_max=4,
        model_name='nomic-embed-text'
    )

    # Inspect results
    print(f"Optimal clusters found: {result_df['Cluster'].nunique()}")
    result_df.to_csv('clustered_reviews.csv', index=False)