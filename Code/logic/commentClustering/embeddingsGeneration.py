import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mean pooling to token embeddings to obtain sentence embeddings.

    Args:
        model_output: Transformer model output with shape [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask with shape [batch_size, seq_len]

    Returns:
        torch.Tensor: Sentence embeddings with shape [batch_size, hidden_size]
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def _load_model_and_tokenizer(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load model and tokenizer from Code/models/ directory.

    Args:
        model_name: Name of the model directory inside Code/models/

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer

    Raises:
        FileNotFoundError: If model path does not exist
        Exception: If there are issues loading the model
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root/"models"/model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at path: {model_path}")

    try:
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info(f"Loading model from {model_path}")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()  # Set to evaluation mode

        logger.info(f"Model loaded successfully on device: {device}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def _encode_texts_batch(texts: List[str], model: AutoModel, tokenizer: AutoTokenizer,
                        batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """
    Encode a list of texts into sentence embeddings using batch processing.

    Args:
        texts: List of texts to encode
        model: Loaded model
        tokenizer: Corresponding tokenizer for the model
        batch_size: Batch size for processing
        max_length: Maximum token length per text

    Returns:
        np.ndarray: Array with sentence embeddings with shape [num_texts, embedding_dim]
    """
    device = next(model.parameters()).device
    all_embeddings = []

    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        # Generate embeddings without computing gradients
        with torch.no_grad():
            model_output = model(**encoded_input)

            # Apply mean pooling to get sentence embeddings
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings (optional but recommended for clustering)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

    return np.vstack(all_embeddings)


def generate_embeddings(texts_df: pd.DataFrame,
                        model_name: str = "nomic-embed-text",
                        batch_size: int = 32,
                        max_length: int = 512) -> pd.DataFrame:
    """
    Generate embeddings for texts in a DataFrame and add them as a new column.

    Args:
        df: DataFrame with a single 'Reviews' column containing text data
        model_name: Name of the model directory in Code/models/ (default: "nomic-embed-text")
        batch_size: Batch size for processing (default: 32)
        max_length: Maximum token length (default: 512)

    Returns:
        pd.DataFrame: DataFrame with original data plus 'embeddings' column

    Raises:
        ValueError: If 'Reviews' column does not exist or DataFrame is empty
        Exception: For other errors during processing
    """
    # Validate input DataFrame
    if texts_df.empty:
        raise ValueError("DataFrame is empty")

    if 'Review' not in texts_df.columns:
        raise ValueError(f"Column 'Review' does not exist in DataFrame. "
                         f"Available columns: {list(texts_df.columns)}")

    # Create a copy to avoid modifying the original DataFrame
    result_df = texts_df.copy()

    try:
        logger.info(f"Processing DataFrame with {len(result_df)} rows")

        # Check and clean null texts
        initial_count = len(result_df)
        result_df = result_df.dropna(subset=['Review'])
        result_df['Review'] = result_df['Review'].astype(str)

        if len(result_df) < initial_count:
            logger.warning(f"Removed {initial_count - len(result_df)} rows with null values in 'Review'")

        if result_df.empty:
            raise ValueError("No valid reviews found after cleaning null values")

        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer(model_name)

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = result_df['Review'].tolist()
        embeddings = _encode_texts_batch(texts, model, tokenizer, batch_size, max_length)

        logger.info(f"Embeddings generated with shape: {embeddings.shape}")

        # Convert embeddings to list for storing in DataFrame
        # Each embedding is saved as JSON string for easy serialization
        embeddings_json = [json.dumps(emb.tolist()) for emb in embeddings]
        result_df['embeddings'] = embeddings_json

        logger.info("Embeddings successfully added to DataFrame")
        return result_df

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


def load_embeddings_from_dataframe(df: pd.DataFrame,
                                   embeddings_column: str = 'embeddings') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract embeddings from a DataFrame column and convert them to numpy array.

    Args:
        df: DataFrame containing embeddings in JSON format
        embeddings_column: Name of column with embeddings (default: 'embeddings')

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: DataFrame with data and array with embeddings

    Raises:
        ValueError: If embeddings column does not exist
    """
    if embeddings_column not in df.columns:
        raise ValueError(f"Column '{embeddings_column}' does not exist in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    try:
        # Convert embeddings from JSON to numpy array
        embeddings = np.array([json.loads(emb) for emb in df[embeddings_column]])

        logger.info(f"Embeddings loaded: {embeddings.shape}")
        return df, embeddings

    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise


def get_available_models() -> List[str]:
    """
    Get list of available models in the Code/models/ directory.

    Returns:
        List[str]: List of available model names
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    models_path = project_root / "models"

    if not models_path.exists():
        logger.warning(f"{models_path} directory does not exist")
        return []

    available_models = [d.name for d in models_path.iterdir() if d.is_dir()]
    logger.info(f"Available models: {available_models}")

    return available_models


# Example usage for testing
if __name__ == "__main__":
    # Example DataFrame creation
    sample_data = {
        'Review': [
            "This product is amazing! I love it.",
            "Not satisfied with the quality.",
            "Great value for money.",
            "Could be better, but it's okay.",
            "Excellent service and fast delivery."
        ]
    }

    df = pd.DataFrame(sample_data)

    try:
        # Check available models
        print("Available models:", get_available_models())

        # Generate embeddings
        df_with_embeddings = generate_embeddings(
            texts_df=df,
            model_name="nomic-embed-text",
            batch_size=16,
            max_length=200
        )

        print(f"DataFrame shape: {df_with_embeddings.shape}")
        print(f"Columns: {list(df_with_embeddings.columns)}")

        # Load embeddings for verification
        df_loaded, embeddings_array = load_embeddings_from_dataframe(df_with_embeddings)
        print(f"Embeddings array shape: {embeddings_array.shape}")

    except Exception as e:
        print(f"Error: {e}")
