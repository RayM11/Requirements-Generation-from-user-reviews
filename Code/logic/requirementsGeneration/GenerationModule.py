import pandas as pd
import re
from typing import Tuple, List, Dict
from Code.logic.requirementsGeneration.LLMManager import LLMManager
from Code.logic.requirementsGeneration.promptTemplates import get_generation_prompt, get_unification_prompt
from Code.logic.requirementsGeneration.RequirementsParser import RequirementsParser


class GenerationModule:
    """
    Module for generating and unifying software requirements from user comments.

    This module processes user comments organized in clusters to generate initial
    requirements per cluster, then unifies them to eliminate redundancy.
    """

    def __init__(self):
        """Initialize the GenerationModule."""
        self.llm_manager = None
        self.parser = RequirementsParser()

    def generate_requirements(
            self,
            reviews_df: pd.DataFrame,
            app_description: str,
            model_name: str,
            provider: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to generate and unify software requirements from user comments.

        Args:
            reviews_df (pd.DataFrame): DataFrame with columns ['ID','Review', 'Cluster']
            app_description (str): Description of the target application
            model_name (str): LLM model name to use
            provider (str): LLM provider to use

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - First DataFrame: Initial requirements by cluster
                - Second DataFrame: Final unified requirements

        DataFrame structures:
            Initial requirements: ['cluster', 'requirement_id', 'type', 'description', 'based_on_comments']
            Unified requirements: ['requirement_id', 'type', 'description', 'based_on_comments']
        """

        # Initialize LLM Manager
        print("Initializing LLM Manager...")
        self.llm_manager = LLMManager(
            model_name=model_name,
            provider=provider,
            test_connection=True
        )

        # Validate input DataFrame
        self._validate_input_dataframe(reviews_df)

        # Step 1: Generate requirements for each cluster
        print("\n" + "=" * 60)
        print("STEP 1: GENERATING REQUIREMENTS BY CLUSTER")
        print("=" * 60)

        initial_requirements_df = self._generate_requirements_by_cluster(
            reviews_df, app_description
        )

        print(
            f"\nGenerated {len(initial_requirements_df)} initial requirements from {reviews_df['Cluster'].nunique()} clusters")

        # Step 2: Unify all requirements
        print("\n" + "=" * 60)
        print("STEP 2: UNIFYING REQUIREMENTS")
        print("=" * 60)

        unified_requirements_df = self._unify_requirements(
            initial_requirements_df, app_description
        )

        print(f"Unified into {len(unified_requirements_df)} final requirements")

        return initial_requirements_df, unified_requirements_df

    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame has the required structure.

        Args:
            df (pd.DataFrame): Input DataFrame to validate

        Raises:
            ValueError: If DataFrame structure is invalid
        """
        required_columns = ['ID', 'Review', 'Cluster']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        if df['ID'].isnull().any():
            raise ValueError("ID column cannot contain null values")

        if df['Review'].isnull().any():
            raise ValueError("Review column cannot contain null values")

        if df['Cluster'].isnull().any():
            raise ValueError("Cluster column cannot contain null values")

    def _generate_requirements_by_cluster(
            self,
            reviews_df: pd.DataFrame,
            app_description: str
    ) -> pd.DataFrame:
        """
        Generate requirements for each cluster of comments.

        Args:
            reviews_df (pd.DataFrame): DataFrame with reviews and clusters
            app_description (str): Application description

        Returns:
            pd.DataFrame: DataFrame with initial requirements by cluster
        """
        all_requirements = []

        # Counters for sequential ID assignment across clusters
        fr_counter = 1
        nfr_counter = 1
        # next_comment_number = 1

        # Process each cluster
        clusters = reviews_df['Cluster'].unique()

        for cluster_id in sorted(clusters):
            print(f"\nProcessing Cluster {cluster_id}...")

            # Get comments for this cluster
            cluster_comments = reviews_df[reviews_df['Cluster'] == cluster_id]['Review'].tolist()
            cluster_comments_ids = reviews_df[reviews_df['Cluster'] == cluster_id]['ID'].tolist()

            # Format comments with numbers for the prompt
            formatted_comments = self._format_comments_for_prompt(cluster_comments, cluster_comments_ids)
            # next_comment_number = next_comment_number + len(cluster_comments)

            # Generate prompt
            prompt = get_generation_prompt(app_description, formatted_comments)

            # Get LLM response
            print(f"  Generating requirements for cluster {cluster_id}...")
            response = self.llm_manager.generate_response(prompt)

            # Parse requirements from response
            cluster_requirements = self.parser.parse_generation_response(response)

            # Assign sequential IDs and add cluster information
            for req in cluster_requirements:
                if req['type'].upper() == 'FUNCTIONAL':
                    req['requirement_id'] = f"FR{fr_counter:03d}"
                    fr_counter += 1
                else:  # NON_FUNCTIONAL
                    req['requirement_id'] = f"NFR{nfr_counter:03d}"
                    nfr_counter += 1

                req['cluster'] = cluster_id
                all_requirements.append(req)

            print(f"  Generated {len(cluster_requirements)} requirements for cluster {cluster_id}")

        # Create DataFrame
        initial_df = pd.DataFrame(all_requirements)

        # Reorder columns
        if not initial_df.empty:
            initial_df = initial_df[['cluster', 'requirement_id', 'type', 'description', 'based_on_comments']]

        return initial_df

    def _unify_requirements(
            self,
            initial_requirements_df: pd.DataFrame,
            app_description: str
    ) -> pd.DataFrame:
        """
        Unify requirements from all clusters to eliminate redundancy.

        Args:
            initial_requirements_df (pd.DataFrame): Initial requirements by cluster
            app_description (str): Application description

        Returns:
            pd.DataFrame: DataFrame with unified requirements
        """
        if initial_requirements_df.empty:
            print("No requirements to unify.")
            return pd.DataFrame(columns=['requirement_id', 'type', 'description', 'based_on_comments'])

        # Format requirements for unification prompt
        formatted_requirements = self._format_requirements_for_unification(initial_requirements_df)

        # Generate unification prompt
        prompt = get_unification_prompt(app_description, formatted_requirements)

        # Get LLM response
        print("Unifying requirements...")
        response = self.llm_manager.generate_response(prompt)

        # Parse unified requirements
        unified_requirements = self.parser.parse_unification_response(response)

        # Create DataFrame
        unified_df = pd.DataFrame(unified_requirements)

        # Reorder columns (no cluster column for unified requirements)
        if not unified_df.empty:
            unified_df = unified_df[['requirement_id', 'type', 'description', 'based_on_comments']]

        return unified_df

    def _format_comments_for_prompt(self, comments: List[str], comments_ids: List[int]) -> str:
        """
        Format comments with sequential numbering for prompt.

        Args:
            comments (List[str]): List of comment texts

        Returns:
            str: Formatted comments string
        """
        formatted_comments = []
        for comment_id, comment in zip(comments_ids, comments):
            formatted_comments.append(f"{comment_id}. {comment}")

        return "\n".join(formatted_comments)

    def _format_requirements_for_unification(self, requirements_df: pd.DataFrame) -> str:
        """
        Format requirements by cluster for the unification prompt.

        Args:
            requirements_df (pd.DataFrame): Requirements DataFrame

        Returns:
            str: Formatted requirements string
        """
        formatted_sections = []

        # Group by cluster
        for cluster_id in sorted(requirements_df['cluster'].unique()):
            cluster_reqs = requirements_df[requirements_df['cluster'] == cluster_id]

            section = f"## CLUSTER {cluster_id} ##\n"

            # Add functional requirements
            fr_reqs = cluster_reqs[cluster_reqs['type'].str.upper() == 'FUNCTIONAL']
            if not fr_reqs.empty:
                for _, req in fr_reqs.iterrows():
                    section += f"**{req['requirement_id']}:** {req['description']} (Based on comments: {req['based_on_comments']})\n"

            # Add non-functional requirements
            nfr_reqs = cluster_reqs[cluster_reqs['type'].str.upper() == 'NON_FUNCTIONAL']
            if not nfr_reqs.empty:
                for _, req in nfr_reqs.iterrows():
                    # Extract NFR type from description if needed
                    nfr_type = self._extract_nfr_type(req['description'])
                    section += f"**{req['requirement_id']} ({nfr_type}):** {req['description']} (Based on comments: {req['based_on_comments']})\n"

            formatted_sections.append(section)

        return "\n".join(formatted_sections)

    def _extract_nfr_type(self, description: str) -> str:
        """
        Extract NFR type from requirement description.

        Args:
            description (str): Requirement description

        Returns:
            str: NFR type (Performance, Usability, Reliability, etc.)
        """
        # Common NFR type keywords
        type_keywords = {
            'performance': ['load', 'speed', 'time', 'fast', 'slow', 'seconds', 'response'],
            'usability': ['user', 'interface', 'navigation', 'ease', 'intuitive', 'accessible'],
            'reliability': ['crash', 'error', 'failure', 'stability', 'available', 'uptime'],
            'security': ['secure', 'authentication', 'authorization', 'password', 'encrypt'],
            'scalability': ['scale', 'concurrent', 'users', 'capacity', 'volume'],
            'maintainability': ['maintain', 'update', 'modify', 'extend', 'documentation']
        }

        description_lower = description.lower()

        for nfr_type, keywords in type_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return nfr_type.capitalize()

        return 'Quality'  # Default type

    def get_summary_statistics(
            self,
            initial_df: pd.DataFrame,
            unified_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Generate summary statistics for the requirements generation process.

        Args:
            initial_df (pd.DataFrame): Initial requirements DataFrame
            unified_df (pd.DataFrame): Unified requirements DataFrame

        Returns:
            Dict: Summary statistics
        """
        stats = {
            'initial_requirements': {
                'total': len(initial_df),
                'functional': len(initial_df[initial_df['type'].str.upper() == 'FUNCTIONAL']),
                'non_functional': len(initial_df[initial_df['type'].str.upper() == 'NON_FUNCTIONAL']),
                'clusters_processed': initial_df['cluster'].nunique() if not initial_df.empty else 0
            },
            'unified_requirements': {
                'total': len(unified_df),
                'functional': len(unified_df[unified_df['type'].str.upper() == 'FUNCTIONAL']),
                'non_functional': len(unified_df[unified_df['type'].str.upper() == 'NON_FUNCTIONAL'])
            },
            'consolidation_ratio': len(unified_df) / len(initial_df) if len(initial_df) > 0 else 0
        }

        return stats

    def print_summary(
            self,
            initial_df: pd.DataFrame,
            unified_df: pd.DataFrame
    ) -> None:
        """
        Print a summary of the requirements generation process.

        Args:
            initial_df (pd.DataFrame): Initial requirements DataFrame
            unified_df (pd.DataFrame): Unified requirements DataFrame
        """
        stats = self.get_summary_statistics(initial_df, unified_df)

        print("\n" + "=" * 60)
        print("REQUIREMENTS GENERATION SUMMARY")
        print("=" * 60)

        print(f"\nInitial Requirements (by cluster):")
        print(f"  Total: {stats['initial_requirements']['total']}")
        print(f"  Functional: {stats['initial_requirements']['functional']}")
        print(f"  Non-functional: {stats['initial_requirements']['non_functional']}")
        print(f"  Clusters processed: {stats['initial_requirements']['clusters_processed']}")

        print(f"\nUnified Requirements:")
        print(f"  Total: {stats['unified_requirements']['total']}")
        print(f"  Functional: {stats['unified_requirements']['functional']}")
        print(f"  Non-functional: {stats['unified_requirements']['non_functional']}")

        print(f"\nConsolidation ratio: {stats['consolidation_ratio']:.2%}")
        print(f"Requirements reduced by: {100 * (1 - stats['consolidation_ratio']):.1f}%")


# Example usage
if __name__ == "__main__":
    """
    Example usage of the GenerationModule.
    """

    # Sample data
    sample_data = {
        'Review': [
            'The app is very slow to load',
            'Cannot edit my profile information',
            'App crashes when uploading files',
            'Loading times are terrible',
            'Need better notification system',
            'Profile editing is broken'
        ],
        'Cluster': [1, 2, 3, 1, 2, 2]
    }

    reviews_df = pd.DataFrame(sample_data)
    initial_reqs = pd.read_csv("initial_requirements.csv")
    app_description = "Task management application for development teams"

    try:
        # Initialize the module
        gen_module = GenerationModule()

        gen_module.llm_manager = LLMManager(
            model_name="deepseek-ai/DeepSeek-V3-0324",
            provider="fireworks-ai",
            test_connection=True
        )

        # # Generate requirements
        # initial_reqs, unified_reqs = gen_module.generate_requirements(
        #     reviews_df=reviews_df,
        #     app_description=app_description,
        #     model_name="deepseek-ai/DeepSeek-V3-0324",
        #     provider="fireworks-ai"
        # )

        unified_reqs = gen_module._unify_requirements(initial_reqs, app_description)

        unified_reqs.to_csv("unified_requirements.csv")

        # Print summary
        gen_module.print_summary(initial_reqs, unified_reqs)

        # Display results
        print("\n" + "=" * 40)
        print("INITIAL REQUIREMENTS")
        print("=" * 40)
        print(initial_reqs.to_string(index=False))

        print("\n" + "=" * 40)
        print("UNIFIED REQUIREMENTS")
        print("=" * 40)
        print(unified_reqs.to_string(index=False))

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure to:")
        print("1. Set your HF_TOKEN environment variable")
        print("2. Install required dependencies")
        print("3. Check your internet connection")
