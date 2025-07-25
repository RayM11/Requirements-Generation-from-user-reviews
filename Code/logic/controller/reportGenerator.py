"""
Report Generator Module for Software Requirements Generation Algorithm

This module provides functions to generate comprehensive reports for each stage
of the software requirements generation process from user opinions.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, Any


def filtering_report(df_original: pd.DataFrame,
                     df_filtered: pd.DataFrame,
                     classification_model: str,
                     vector_type: str,
                     save_path: str) -> None:
    """
    Generate a filtering report comparing original and filtered datasets.

    This function creates a comprehensive report of the filtering stage,
    documenting the classification model used, vector type, and the impact
    of filtering on the dataset size.

    Args:
        df_original (pd.DataFrame): Original dataset with 'Review' column
        df_filtered (pd.DataFrame): Filtered dataset with 'Review' column
        classification_model (str): Name of the classification model used
        vector_type (str): Type of vector representation used
        save_path (str): Directory path where the report will be saved

    Returns:
        None

    Raises:
        ValueError: If dataframes don't contain 'Review' column
        OSError: If save_path directory doesn't exist or isn't writable
    """

    # Validate input dataframes
    if 'Review' not in df_original.columns:
        raise ValueError("Original dataframe must contain 'Review' column")
    if 'Review' not in df_filtered.columns:
        raise ValueError("Filtered dataframe must contain 'Review' column")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Calculate statistics
    original_count = len(df_original)
    filtered_count = len(df_filtered)
    remaining_percentage = (filtered_count / original_count) * 100 if original_count > 0 else 0

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create report content
    report_content = f"""
FILTERING STAGE REPORT
======================

Report Generated: {timestamp}

CONFIGURATION
-------------
Classification Model: {classification_model}
Vector Type: {vector_type}

FILTERING RESULTS
-----------------
Original Comments Count: {original_count:,}
Comments After Filtering: {filtered_count:,}
Remaining Comments Percentage: {remaining_percentage:.2f}%

SUMMARY
-------
The filtering process removed {original_count - filtered_count:,} comments ({100 - remaining_percentage:.2f}% of original dataset).
This indicates that {remaining_percentage:.2f}% of the original comments were considered relevant for further processing.

"""

    # Save report
    report_filename = os.path.join(save_path, "filtering_report.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Filtering report saved to: {report_filename}")


def clustering_report(df_clustered: pd.DataFrame,
                      clustering_algorithm: str,
                      reduction_algorithm: str,
                      save_path: str) -> None:
    """
    Generate a clustering report and save the clustered dataset.

    This function creates a report of the clustering stage, documenting
    the algorithms used and the distribution of comments across clusters.

    Args:
        df_clustered (pd.DataFrame): Dataset with 'Review' and 'Cluster' columns
        clustering_algorithm (str): Name of the clustering algorithm used
        reduction_algorithm (str): Name of the dimensionality reduction algorithm used
        save_path (str): Directory path where files will be saved

    Returns:
        None

    Raises:
        ValueError: If dataframe doesn't contain required columns
        OSError: If save_path directory doesn't exist or isn't writable
    """

    # Validate input dataframe
    required_columns = ['Review', 'Cluster']
    missing_columns = [col for col in required_columns if col not in df_clustered.columns]
    if missing_columns:
        raise ValueError(f"Dataframe must contain columns: {missing_columns}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save clustered dataset
    csv_filename = os.path.join(save_path, "clustered_data.csv")
    df_clustered.to_csv(csv_filename, index=False, encoding='utf-8')

    # Calculate clustering statistics
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    total_clusters = len(cluster_counts)
    total_comments = len(df_clustered)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create cluster distribution text
    distribution_text = []
    for cluster_id in sorted(cluster_counts.index):
        count = cluster_counts[cluster_id]
        percentage = (count / total_comments) * 100
        distribution_text.append(f"Cluster {cluster_id}: {count} comments ({percentage:.1f}%)")

    # Create report content
    report_content = f"""
CLUSTERING STAGE REPORT
=======================

Report Generated: {timestamp}

CONFIGURATION
-------------
Clustering Algorithm: {clustering_algorithm}
Reduction Algorithm: {reduction_algorithm}

CLUSTERING RESULTS
------------------
Total Clusters Identified: {total_clusters}
Total Comments Processed: {total_comments:,}

CLUSTER DISTRIBUTION
-------------------
{chr(10).join(distribution_text)}

SUMMARY
-------
The clustering process successfully grouped {total_comments:,} comments into {total_clusters} distinct clusters.
Average cluster size: {total_comments / total_clusters:.1f} comments per cluster.

FILES GENERATED
---------------
- Clustered dataset: clustered_data.csv
- This report: clustering_report.txt

"""

    # Save report
    report_filename = os.path.join(save_path, "clustering_report.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Clustering report saved to: {report_filename}")
    print(f"Clustered dataset saved to: {csv_filename}")


def generation_report(df_initial: pd.DataFrame,
                      df_unified: pd.DataFrame,
                      llm_model: str,
                      provider: str,
                      save_path: str) -> None:
    """
    Generate a requirements generation report and save both datasets.

    This function creates a comprehensive report of the requirements generation stage,
    documenting the LLM used, provider, and the impact of the unification process.

    Args:
        df_initial (pd.DataFrame): Initial requirements with columns:
                                 ['cluster', 'requirement_id', 'type', 'description', 'based_on_comments']
        df_unified (pd.DataFrame): Unified requirements with columns:
                                 ['requirement_id', 'type', 'description', 'based_on_comments']
        llm_model (str): Name of the LLM model used
        provider (str): Name of the LLM provider
        save_path (str): Directory path where files will be saved

    Returns:
        None

    Raises:
        ValueError: If dataframes don't contain required columns
        OSError: If save_path directory doesn't exist or isn't writable
    """

    # Validate input dataframes
    initial_required_cols = ['cluster', 'requirement_id', 'type', 'description', 'based_on_comments']
    unified_required_cols = ['requirement_id', 'type', 'description', 'based_on_comments']

    initial_missing = [col for col in initial_required_cols if col not in df_initial.columns]
    unified_missing = [col for col in unified_required_cols if col not in df_unified.columns]

    if initial_missing:
        raise ValueError(f"Initial requirements dataframe missing columns: {initial_missing}")
    if unified_missing:
        raise ValueError(f"Unified requirements dataframe missing columns: {unified_missing}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save both datasets
    initial_csv = os.path.join(save_path, "initial_requirements.csv")
    unified_csv = os.path.join(save_path, "unified_requirements.csv")

    df_initial.to_csv(initial_csv, index=False, encoding='utf-8')
    df_unified.to_csv(unified_csv, index=False, encoding='utf-8')

    # Calculate statistics for initial requirements
    initial_functional = len(df_initial[df_initial['type'] == 'FUNCTIONAL'])
    initial_non_functional = len(df_initial[df_initial['type'] == 'NON_FUNCTIONAL'])
    initial_total = len(df_initial)

    # Calculate statistics for unified requirements
    unified_functional = len(df_unified[df_unified['type'] == 'FUNCTIONAL'])
    unified_non_functional = len(df_unified[df_unified['type'] == 'NON_FUNCTIONAL'])
    unified_total = len(df_unified)

    # Calculate eliminated requirements
    eliminated_functional = initial_functional - unified_functional
    eliminated_non_functional = initial_non_functional - unified_non_functional
    eliminated_total = initial_total - unified_total

    # Calculate reduction percentage
    reduction_percentage = (eliminated_total / initial_total) * 100 if initial_total > 0 else 0

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create report content
    report_content = f"""
REQUIREMENTS GENERATION STAGE REPORT
====================================

Report Generated: {timestamp}

CONFIGURATION
-------------
LLM Model: {llm_model}
Provider: {provider}

INITIAL REQUIREMENTS (BY CLUSTER)
---------------------------------
Functional Requirements: {initial_functional}
Non-Functional Requirements: {initial_non_functional}
Total Initial Requirements: {initial_total}

UNIFIED REQUIREMENTS (AFTER DEDUPLICATION)
------------------------------------------
Functional Requirements: {unified_functional}
Non-Functional Requirements: {unified_non_functional}
Total Unified Requirements: {unified_total}

UNIFICATION IMPACT
------------------
Eliminated Requirements:
  - Functional: {eliminated_functional}
  - Non-Functional: {eliminated_non_functional}
  - Total: {eliminated_total}

Reduction Percentage: {reduction_percentage:.2f}%

SUMMARY
-------
The requirements generation process initially produced {initial_total} requirements across all clusters.
After the unification process to eliminate redundancy, {unified_total} unique requirements remained.
This represents a {reduction_percentage:.2f}% reduction, indicating {'high' if reduction_percentage > 30 else 'moderate' if reduction_percentage > 10 else 'low'} redundancy in the initial set.

EFFICIENCY METRICS
------------------
Requirements Retained: {100 - reduction_percentage:.2f}%
Functional Requirements Retention: {(unified_functional / initial_functional * 100) if initial_functional > 0 else 0:.1f}%
Non-Functional Requirements Retention: {(unified_non_functional / initial_non_functional * 100) if initial_non_functional > 0 else 0:.1f}%

FILES GENERATED
---------------
- Initial requirements: initial_requirements.csv
- Unified requirements: unified_requirements.csv
- This report: generation_report.txt

"""

    # Save report
    report_filename = os.path.join(save_path, "generation_report.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Generation report saved to: {report_filename}")
    print(f"Initial requirements saved to: {initial_csv}")
    print(f"Unified requirements saved to: {unified_csv}")


def generate_final_summary(results: Dict[str, Any], save_path: str) -> None:
        """Generate a final summary report of the entire pipeline."""

        # Build cluster distribution text
        cluster_text = ""
        for cluster_id, count in sorted(results['clustering']['cluster_distribution'].items()):
            percentage = (count / results['clustering']['comments_clustered']) * 100 if results['clustering']['comments_clustered'] > 0 else 0
            cluster_text += f"   Cluster {cluster_id}: {count} comments ({percentage:.1f}%)\n"

        summary_content = f"""
REQUIREMENTS GENERATION PIPELINE SUMMARY
========================================

Execution Time: {results['metadata']['start_time']} to {results['metadata']['end_time']}
Duration: {results['metadata']['duration']}
Output Directory: {results['metadata']['run_directory']}

PIPELINE OVERVIEW
-----------------

STAGE RESULTS
-------------

1. FILTERING STAGE
   Original Comments: {results['filtering']['original_count']:,}
   Filtered Comments: {results['filtering']['filtered_count']:,}
   Retention Rate: {results['filtering']['retention_rate']:.2%}

2. CLUSTERING STAGE
   Total Clusters: {results['clustering']['total_clusters']}
   Comments Clustered: {results['clustering']['comments_clustered']:,}

   Cluster Distribution:
{cluster_text}

3. GENERATION STAGE
   Initial Requirements: {results['generation']['initial_requirements']}
   Unified Requirements: {results['generation']['unified_requirements']}
   Consolidation Ratio: {results['generation']['consolidation_ratio']:.2%}

   Final Requirements Breakdown:
   - Functional: {results['generation']['functional_count']}
   - Non-Functional: {results['generation']['non_functional_count']}

CONFIGURATION USED
------------------
Filtering:
  - Model: {results['metadata']['parameters']['filtering']['classification_model']}
  - Vector Type: {results['metadata']['parameters']['filtering']['vector_type']}

Clustering:
  - Algorithm: {results['metadata']['parameters']['clustering']['clustering_algorithm']}
  - Dimensionality Reduction: {results['metadata']['parameters']['clustering']['dim_reduction']}
  - K Range: {results['metadata']['parameters']['clustering']['k_min']}-{results['metadata']['parameters']['clustering']['k_max']}

Generation:
  - LLM Model: {results['metadata']['parameters']['generation']['llm_model']}
  - Provider: {results['metadata']['parameters']['generation']['llm_provider']}

FILES GENERATED
---------------
- filtering_report.txt
- clustering_report.txt  
- clustered_data.csv
- generation_report.txt
- initial_requirements.csv
- unified_requirements.csv
- pipeline_summary.txt
"""

        # Save summary
        summary_path = os.path.join(save_path, "pipeline_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        print(f"Pipeline summary saved to: {summary_path}")


# Example usage and testing functions
def _test_functions():
    """
    Test function to demonstrate usage of all report functions.
    This function is for development/testing purposes only.
    """

    # Create sample data for testing
    import numpy as np

    # Sample data for filtering test
    df_orig = pd.DataFrame({
        'Review': ['Great app', 'Terrible service', 'Love it', 'Hate this', 'Perfect']
    })
    df_filt = pd.DataFrame({
        'Review': ['Great app', 'Love it', 'Perfect']
    })

    # Sample data for clustering test
    df_clust = pd.DataFrame({
        'Review': ['Great app', 'Love it', 'Perfect', 'Amazing', 'Excellent'],
        'Cluster': [0, 0, 1, 1, 1]
    })

    # Sample data for generation test
    df_init = pd.DataFrame({
        'cluster': [0, 0, 1, 1, 2],
        'requirement_id': ['R001', 'R002', 'R003', 'R004', 'R005'],
        'type': ['FUNCTIONAL', 'NON_FUNCTIONAL', 'FUNCTIONAL', 'FUNCTIONAL', 'NON_FUNCTIONAL'],
        'description': ['Req 1', 'Req 2', 'Req 3', 'Req 4', 'Req 5'],
        'based_on_comments': ['Comment 1', 'Comment 2', 'Comment 3', 'Comment 4', 'Comment 5']
    })

    df_unif = pd.DataFrame({
        'requirement_id': ['R001', 'R003', 'R005'],
        'type': ['FUNCTIONAL', 'FUNCTIONAL', 'NON_FUNCTIONAL'],
        'description': ['Req 1', 'Req 3', 'Req 5'],
        'based_on_comments': ['Comment 1', 'Comment 3', 'Comment 5']
    })

    # Test all functions
    test_path = "./test_reports"

    print("Testing filtering report...")
    filtering_report(df_orig, df_filt, "RandomForest", "TF-IDF", test_path)

    print("\nTesting clustering report...")
    clustering_report(df_clust, "KMeans", "PCA", test_path)

    print("\nTesting generation report...")
    generation_report(df_init, df_unif, "GPT-4", "OpenAI", test_path)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    # Uncomment the line below to run tests
    _test_functions()