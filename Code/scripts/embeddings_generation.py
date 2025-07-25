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


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load BERTweet model and tokenizer from local path.

    Args:
        model_path: Path to directory containing the BERTweet model

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer

    Raises:
        FileNotFoundError: If model path does not exist
        Exception: If there are issues loading the model
    """
    model_path = Path(model_path)

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


def encode_texts_batch(texts: List[str], model: AutoModel, tokenizer: AutoTokenizer,
                       batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """
    Encode a list of texts into sentence embeddings using batch processing.

    Args:
        texts: List of texts to encode
        model: Loaded BERTweet model
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


def generate_embeddings_csv(csv_path: str, model_path: str,
                            text_column: str = 'Review',
                            batch_size: int = 32,
                            max_length: int = 512,
                            output_suffix: str = '_with_embeddings') -> str:
    """
    Generate embeddings for texts in a CSV and create a new file with embeddings added.

    Args:
        csv_path: Path to CSV file with data
        model_path: Path to BERTweet model directory
        text_column: Name of column containing text (default: 'Review')
        batch_size: Batch size for processing (default: 32)
        max_length: Maximum token length (default: 512)
        output_suffix: Suffix for output file (default: '_with_embeddings')

    Returns:
        str: Path to generated CSV file with embeddings

    Raises:
        FileNotFoundError: If CSV file is not found
        ValueError: If specified column does not exist
        Exception: For other errors during processing
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        # Load dataset
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' does not exist in dataset. "
                             f"Available columns: {list(df.columns)}")

        logger.info(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

        # Check and clean null texts
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df[text_column] = df[text_column].astype(str)

        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with null values in '{text_column}'")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = df[text_column].tolist()
        embeddings = encode_texts_batch(texts, model, tokenizer, batch_size, max_length)

        logger.info(f"Embeddings generated with shape: {embeddings.shape}")

        # Convert embeddings to list for storing in CSV
        # Each embedding is saved as JSON string for easy reading later
        embeddings_json = [json.dumps(emb.tolist()) for emb in embeddings]
        df['embeddings'] = embeddings_json

        # Create output file
        output_path = csv_path.parent / f"{csv_path.stem}{output_suffix}{csv_path.suffix}"

        logger.info(f"Saving results to {output_path}")
        df.to_csv(output_path, index=False)

        logger.info(f"Process completed successfully. Generated file: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


def load_embeddings_from_csv(csv_path: str, embeddings_column: str = 'embeddings') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load embeddings from a previously generated CSV.

    Args:
        csv_path: Path to CSV file with embeddings
        embeddings_column: Name of column with embeddings (default: 'embeddings')

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: DataFrame with data and array with embeddings
    """
    df = pd.read_csv(csv_path)

    if embeddings_column not in df.columns:
        raise ValueError(f"Column '{embeddings_column}' does not exist in dataset")

    # Convert embeddings from JSON to numpy array
    embeddings = np.array([json.loads(emb) for emb in df[embeddings_column]])

    logger.info(f"Embeddings loaded: {embeddings.shape}")
    return df, embeddings


# Example usage
if __name__ == "__main__":
    # Example configuration
    csv_file = "../data/transfermovil_informative.csv"  # Replace with actual path
    model_directory = "../models/nomic-embed-text"  # Replace with actual model path

    try:
        # Generate embeddings
        output_file = generate_embeddings_csv(
            csv_path=csv_file,
            model_path=model_directory,
            text_column='Review',
            batch_size=16,  # Adjust based on available memory
            max_length=256,  # Adjust based on typical text length
            output_suffix="with_embeddings (nomic)"
        )

        print(f"Embeddings generated successfully at: {output_file}")

        # Load embeddings for verification
        df, embeddings = load_embeddings_from_csv(output_file)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"First 5 rows of dataset:")
        print(df.head())

    except Exception as e:
        print(f"Error: {e}")
