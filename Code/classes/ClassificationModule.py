import torch
import pandas as pd
from transformers import AutoTokenizer
from CommentFilter import CommentFilter
from tqdm import tqdm
from Code.classes import DataControlModule


def load_model(model_path, model_name='roberta-base', device=None):
    """Load the pre-trained model.

    Args:
        model_path (str): Path to the .pth model file
        model_name (str, optional): Name of the original pre-trained model.
            Defaults to 'roberta-base'.
        device (str, optional): Device to load the model on (cuda/cpu).
            Defaults to cuda if available, otherwise cpu.

    Returns:
        tuple: Loaded model and configured tokenizer
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration needed to initialize the model
    config = {
        'model': model_name,
        'lr': 2e-5,
        'weight_decay': 0.01,
        'warmup': 0.1,
        'batch_size': 16  # Default value, doesn't affect inference
    }

    # Initialize model and load weights
    model = CommentFilter(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def classify_comments(df, model, tokenizer, batch_size=16, max_length=200, device=None):
    """Classify comments using the loaded model.

    Args:
        df (pandas.DataFrame): DataFrame with comments
        model (CommentFilter): Pre-trained model
        tokenizer: Configured tokenizer
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        device (str, optional): Device to run the model on.
            Defaults to cuda if available, otherwise cpu.

    Returns:
        pandas.DataFrame: DataFrame with comments and their classifications
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # List to store predictions
    predictions = []

    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Classifying comments"):
        batch = df['comment'][i:i + batch_size].tolist()

        # Tokenize comments
        inputs = tokenizer(
            batch,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Make prediction
        with torch.no_grad():
            _, logits = model(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'])

            # Convert logits to binary predictions
            batch_predictions = (logits > 0.5).int().cpu().numpy()

        # Add predictions to the list
        predictions.extend(batch_predictions.flatten().tolist())

    # Add predictions as a new column
    df['is_informative'] = predictions

    return df


def classify_comments_from_file(csv_path, model_path, output_path=None,
                                model_name='roberta-base',
                                batch_size=32, max_length=128):
    """Complete process to classify comments from a file.

    Args:
        csv_path (str): Path to the CSV file with comments
        model_path (str): Path to the .pth model file
        output_path (str, optional): Path to save results.
            Defaults to 'classified_comments.csv' if None.
        model_name (str, optional): Name of the original pre-trained model.
            Defaults to 'roberta-base'.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        max_length (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        pandas.DataFrame: DataFrame with comments and their classifications
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = 'classified_comments.csv'

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    model, tokenizer = load_model(model_path, model_name, device)

    # Load comments
    df = DataControlModule.load_comments(csv_path)

    # Classify comments
    df = classify_comments(df, model, tokenizer, batch_size, max_length, device)

    # Save results
    df = DataControlModule.save_results(df, output_path)

    return df


def get_classification_summary(df):
    """Get a summary of the classification results.

    Args:
        df (pandas.DataFrame): DataFrame with classified comments

    Returns:
        dict: Dictionary with summary statistics
    """
    n_informative = df['is_informative'].sum()
    total = len(df)
    percentage = (n_informative / total) * 100 if total > 0 else 0

    return {
        'total_comments': total,
        'informative_comments': int(n_informative),
        'non_informative_comments': total - int(n_informative),
        'informative_percentage': percentage
    }