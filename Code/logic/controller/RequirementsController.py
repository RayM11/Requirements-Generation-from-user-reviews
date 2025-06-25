"""
Requirements Controller Module

This module orchestrates the complete requirements generation algorithm from user comments.
It implements the Singleton pattern and coordinates three main phases:
1. Comment Filtering using ClassificationModule
2. Comment Clustering using ClusteringModule
3. Requirements Generation using GenerationModule

Each phase generates comprehensive reports using the reportGenerator functions.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Import the processing modules
from Code.logic.commentFiltering.FilteringModule import FilteringModule
from Code.logic.commentClustering.ClusteringModule import ClusteringModule
from Code.logic.requirementsGeneration.GenerationModule import GenerationModule

# Import report generation functions
from Code.logic.controller.reportGenerator import filtering_report, clustering_report, generation_report, generate_final_summary


class RequirementsController:
    """
    Singleton controller for orchestrating the requirements generation pipeline.

    This controller manages the complete flow from raw user comments to structured
    software requirements through filtering, clustering, and generation phases.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Implement Singleton pattern - only one controller instance allowed."""
        if cls._instance is None:
            cls._instance = super(RequirementsController, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the controller (only once due to Singleton pattern)."""
        if not self._initialized:
            self.filtering_module = FilteringModule()
            self.clustering_module = ClusteringModule()
            self.generation_module = GenerationModule()
            self.run_directory = None
            self.start_time = None
            self._initialized = True

    def generate_requirements_from_csv(
        self,
        csv_path: str,
        app_description: str,
        output_directory: str,
        # Filtering parameters
        classification_model: str = "BERTweet - base",
        vector_type: str = "None",
        # Clustering parameters
        clustering_algorithm: str = 'k-means',
        dim_reduction: str = 'None',
        k_min: int = 3,
        k_max: int = 10,
        embedding_model: str = "nomic-embed-text",
        batch_size: int = 32,
        max_length: int = 512,
        reduction_kwargs: Optional[Dict] = None,
        clustering_kwargs: Optional[Dict] = None,
        # Generation parameters
        llm_model: str = "deepseek-ai/DeepSeek-V3-0324",
        llm_provider: str = "fireworks-ai"
    ) -> Dict[str, Any]:
        """
        Execute the complete requirements generation pipeline.

        Args:
            csv_path (str): Path to CSV file containing user comments in 'Review' column
            app_description (str): Description of the target application
            output_directory (str): Directory where reports and outputs will be saved

            # Filtering parameters
            classification_model (str): Classification model name
            vector_type (str): Vector type for filtering ("None", "RC", "RP")

            # Clustering parameters
            clustering_algorithm (str): Clustering algorithm ('k-means', 'fuzzy', 'agglomerative')
            dim_reduction (str): Dimensionality reduction method ('None', 'PCA', 'UMAP')
            k_min (int): Minimum number of clusters to try
            k_max (int): Maximum number of clusters to try
            embedding_model (str): Name of the embedding model
            batch_size (int): Batch size for embedding generation
            max_length (int): Maximum sequence length for tokenization
            reduction_kwargs (Dict): Additional arguments for dimensionality reduction
            clustering_kwargs (Dict): Additional arguments for clustering

            # Generation parameters
            llm_model (str): LLM model name
            llm_provider (str): LLM provider name

        Returns:
            Dict[str, Any]: Complete results and metadata from all pipeline phases

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV doesn't contain 'Review' column or is empty
            RuntimeError: If any pipeline phase fails
        """

        # Initialize pipeline
        self._initialize_pipeline(csv_path, output_directory, app_description)

        try:
            # Load and validate input data
            print("Loading input data...")
            original_df = self._load_and_validate_csv(csv_path)
            print(f"Loaded {len(original_df)} comments from CSV")

            # Phase 1: Filtering
            print("\n" + "="*80)
            print("PHASE 1: COMMENT FILTERING")
            print("="*80)

            filtered_df = self._execute_filtering_phase(
                original_df, classification_model, vector_type
            )

            # Generate filtering report
            filtering_report(
                df_original=original_df,
                df_filtered=filtered_df,
                classification_model=classification_model,
                vector_type=vector_type,
                save_path=self.run_directory
            )

            # Phase 2: Clustering
            print("\n" + "="*80)
            print("PHASE 2: COMMENT CLUSTERING")
            print("="*80)

            clustered_df = self._execute_clustering_phase(
                filtered_df,
                clustering_algorithm,
                dim_reduction,
                k_min,
                k_max,
                embedding_model,
                batch_size,
                max_length,
                reduction_kwargs,
                clustering_kwargs
            )

            # Generate clustering report
            clustering_report(
                df_clustered=clustered_df,
                clustering_algorithm=clustering_algorithm,
                reduction_algorithm=dim_reduction,
                save_path=self.run_directory
            )

            # Phase 3: Requirements Generation
            print("\n" + "="*80)
            print("PHASE 3: REQUIREMENTS GENERATION")
            print("="*80)

            initial_requirements_df, unified_requirements_df = self._execute_generation_phase(
                clustered_df, app_description, llm_model, llm_provider
            )

            # Generate generation report
            generation_report(
                df_initial=initial_requirements_df,
                df_unified=unified_requirements_df,
                llm_model=llm_model,
                provider=llm_provider,
                save_path=self.run_directory
            )

            # Generate final summary
            results = self._compile_results(
                original_df, filtered_df, clustered_df,
                initial_requirements_df, unified_requirements_df,
                {
                    'filtering': {
                        'classification_model': classification_model,
                        'vector_type': vector_type
                    },
                    'clustering': {
                        'clustering_algorithm': clustering_algorithm,
                        'dim_reduction': dim_reduction,
                        'k_min': k_min,
                        'k_max': k_max,
                        'embedding_model': embedding_model
                    },
                    'generation': {
                        'llm_model': llm_model,
                        'llm_provider': llm_provider
                    }
                }
            )

            generate_final_summary(results, save_path=self.run_directory)

            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Results saved to: {self.run_directory}")

            return results

        except Exception as e:
            print(f"\nPIPELINE FAILED: {str(e)}")
            raise RuntimeError(f"Requirements generation pipeline failed: {str(e)}")

    def _initialize_pipeline(self, csv_path: str, output_directory: str, app_description: str):
        """Initialize pipeline with timestamp and directories."""
        self.start_time = datetime.now()

        # Create timestamped run directory
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.run_directory = os.path.join(output_directory, f"requirements_run_{timestamp}")
        os.makedirs(self.run_directory, exist_ok=True)

        print(f"Pipeline initialized at: {self.start_time}")
        print(f"Output directory: {self.run_directory}")
        print(f"Processing CSV: {csv_path}")
        print(f"Application: {app_description}")

    def _load_and_validate_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the input CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

        if 'Review' not in df.columns:
            raise ValueError("CSV file must contain a 'Review' column")

        if df.empty:
            raise ValueError("CSV file is empty")

        # Remove rows with null/empty reviews
        original_count = len(df)
        df = df.dropna(subset=['Review'])
        df = df[df['Review'].str.strip() != '']

        if len(df) == 0:
            raise ValueError("No valid reviews found in CSV file")

        if len(df) < original_count:
            print(f"Removed {original_count - len(df)} empty/null reviews")

        return df.reset_index(drop=True)

    def _execute_filtering_phase(
        self,
        original_df: pd.DataFrame,
        classification_model: str,
        vector_type: str
    ) -> pd.DataFrame:
        """Execute the comment filtering phase."""
        print(f"Filtering {len(original_df)} comments...")
        print(f"Model: {classification_model}")
        print(f"Vector type: {vector_type}")

        try:
            filtered_df = self.filtering_module.filter_comments(
                dataframe=original_df,
                base_model=classification_model,
                knowledge=vector_type
            )

            print(f"Filtering complete: {len(filtered_df)} relevant comments retained")
            print(f"Retention rate: {len(filtered_df)/len(original_df)*100:.1f}%")

            return filtered_df

        except Exception as e:
            raise RuntimeError(f"Filtering phase failed: {str(e)}")

    def _execute_clustering_phase(
        self,
        filtered_df: pd.DataFrame,
        clustering_algorithm: str,
        dim_reduction: str,
        k_min: int,
        k_max: int,
        embedding_model: str,
        batch_size: int,
        max_length: int,
        reduction_kwargs: Optional[Dict],
        clustering_kwargs: Optional[Dict]
    ) -> pd.DataFrame:
        """Execute the comment clustering phase."""
        print(f"Clustering {len(filtered_df)} comments...")
        print(f"Algorithm: {clustering_algorithm}")
        print(f"Dimensionality reduction: {dim_reduction}")
        print(f"K range: {k_min}-{k_max}")

        try:
            clustered_df = self.clustering_module.cluster_reviews(
                df=filtered_df,
                clustering_algorithm=clustering_algorithm,
                dim_reduction=dim_reduction,
                k_min=k_min,
                k_max=k_max,
                model_name=embedding_model,
                batch_size=batch_size,
                max_length=max_length,
                reduction_kwargs=reduction_kwargs,
                clustering_kwargs=clustering_kwargs
            )

            num_clusters = clustered_df['Cluster'].nunique()
            print(f"Clustering complete: {num_clusters} clusters identified")

            # Print cluster distribution
            cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
            print("Cluster distribution:")
            for cluster_id, count in cluster_counts.items():
                percentage = (count / len(clustered_df)) * 100
                print(f"  Cluster {cluster_id}: {count} comments ({percentage:.1f}%)")

            return clustered_df

        except Exception as e:
            raise RuntimeError(f"Clustering phase failed: {str(e)}")

    def _execute_generation_phase(
        self,
        clustered_df: pd.DataFrame,
        app_description: str,
        llm_model: str,
        llm_provider: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the requirements generation phase."""
        print(f"Generating requirements from {clustered_df['Cluster'].nunique()} clusters...")
        print(f"LLM Model: {llm_model}")
        print(f"Provider: {llm_provider}")

        try:
            initial_requirements_df, unified_requirements_df = self.generation_module.generate_requirements(
                reviews_df=clustered_df,
                app_description=app_description,
                model_name=llm_model,
                provider=llm_provider
            )

            print(f"Generation complete:")
            print(f"  Initial requirements: {len(initial_requirements_df)}")
            print(f"  Unified requirements: {len(unified_requirements_df)}")

            # Print requirements breakdown
            if not unified_requirements_df.empty:
                functional_count = len(unified_requirements_df[unified_requirements_df['type'].str.upper() == 'FUNCTIONAL'])
                non_functional_count = len(unified_requirements_df[unified_requirements_df['type'].str.upper() == 'NON_FUNCTIONAL'])
                print(f"  Functional: {functional_count}")
                print(f"  Non-functional: {non_functional_count}")

            return initial_requirements_df, unified_requirements_df

        except Exception as e:
            raise RuntimeError(f"Generation phase failed: {str(e)}")

    def _compile_results(
        self,
        original_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        clustered_df: pd.DataFrame,
        initial_requirements_df: pd.DataFrame,
        unified_requirements_df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive results from all pipeline phases."""
        end_time = datetime.now()

        # Calculate cluster distribution
        cluster_distribution = {}
        if not clustered_df.empty:
            cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
            cluster_distribution = {int(k): int(v) for k, v in cluster_counts.items()}

        # Calculate requirements breakdown
        functional_count = 0
        non_functional_count = 0
        if not unified_requirements_df.empty:
            functional_count = len(unified_requirements_df[unified_requirements_df['type'].str.upper() == 'FUNCTIONAL'])
            non_functional_count = len(unified_requirements_df[unified_requirements_df['type'].str.upper() == 'NON_FUNCTIONAL'])

        results = {
            'metadata': {
                'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': str(end_time - self.start_time),
                'run_directory': self.run_directory,
                'parameters': parameters
            },
            'filtering': {
                'original_count': len(original_df),
                'filtered_count': len(filtered_df),
                'retention_rate': len(filtered_df) / len(original_df) if len(original_df) > 0 else 0
            },
            'clustering': {
                'total_clusters': clustered_df['Cluster'].nunique() if not clustered_df.empty else 0,
                'comments_clustered': len(clustered_df),
                'cluster_distribution': cluster_distribution
            },
            'generation': {
                'initial_requirements': len(initial_requirements_df),
                'unified_requirements': len(unified_requirements_df),
                'functional_count': functional_count,
                'non_functional_count': non_functional_count,
                'consolidation_ratio': len(unified_requirements_df) / len(initial_requirements_df) if len(initial_requirements_df) > 0 else 0
            }
        }

        return results

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the controller."""
        return cls()

    def reset_instance(self):
        """Reset the singleton instance (for testing purposes)."""
        RequirementsController._instance = None
        RequirementsController._initialized = False


# Example usage
if __name__ == "__main__":
    """
    Example usage of the RequirementsController.
    """

    # Get the singleton controller instance
    controller = RequirementsController.get_instance()

    # Example parameters
    csv_path = "user_comments.csv"  # Path to your CSV file
    app_description = "Task management application for development teams"
    output_directory = "./output"

    try:
        # Execute the complete pipeline
        results = controller.generate_requirements_from_csv(
            csv_path=csv_path,
            app_description=app_description,
            output_directory=output_directory,
            # Filtering parameters
            classification_model="BERTweet - base",
            vector_type="RC",
            # Clustering parameters
            clustering_algorithm='k-means',
            dim_reduction='PCA',
            k_min=3,
            k_max=8,
            # Generation parameters
            llm_model="deepseek-ai/DeepSeek-V3-0324",
            llm_provider="fireworks-ai"
        )

        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Original comments: {results['filtering']['original_count']}")
        print(f"Filtered comments: {results['filtering']['filtered_count']}")
        print(f"Clusters found: {results['clustering']['total_clusters']}")
        print(f"Final requirements: {results['generation']['unified_requirements']}")
        print(f"Results saved to: {results['metadata']['run_directory']}")

    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        print("\nMake sure to:")
        print("1. Provide a valid CSV file with 'Review' column")
        print("2. Set required environment variables (API keys, etc.)")
        print("3. Install all required dependencies")
        print("4. Check model paths and availability")
        