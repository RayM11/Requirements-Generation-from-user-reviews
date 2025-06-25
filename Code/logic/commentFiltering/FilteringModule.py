from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import warnings
import os

from Code.logic.commentFiltering.CommentDataset import CommentDataset
from Code.logic.commentFiltering.CommentDatasetFV import CommentDatasetFV
from Code.logic.commentFiltering.knowledgeInjection import collate_function


class FilteringModule:
    """
    Unified interface for filtering comments using pre-trained models
    with or without domain-specific feature vectors.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.knowledge_type = None
        self.max_token_len = 130  # Default value based on your code
        self.batch_size = 16  # Default value
        self.device = self._setup_device()

    def _setup_device(self):
        """Setup computing device (GPU if available, otherwise CPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device

    def filter_comments(self, dataframe, base_model, knowledge="None"):
        """
        Main function for filtering comments.

        Args:
            dataframe (pd.DataFrame): DataFrame with a 'Review' column
            base_model (str): Base model name (e.g., "BERTweet - base")
            knowledge (str): Vector type ("None", "RC", "RP")

        Returns:
            pd.DataFrame: Filtered DataFrame with only relevant comments (class 1)
        """
        # Validate input
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("First argument must be a pandas DataFrame")

        if 'Review' not in dataframe.columns:
            raise ValueError("DataFrame must contain a column named 'Review'")

        if knowledge not in ["None", "RC", "RP"]:
            raise ValueError("Parameter 'conocimiento' must be 'None', 'RC' or 'RP'")

        # Load model and tokenizer
        self._load_model_and_tokenizer(base_model, knowledge)

        # Prepare data for prediction
        prediction_data = self._prepare_prediction_data(dataframe)

        # Create dataset and dataloader
        dataset = self._create_dataset(prediction_data)
        dataloader = self._create_dataloader(dataset)

        # Make predictions
        predictions = self._predict(dataloader)

        # Filter and return results
        return self._filter_results(dataframe, predictions)

    def _load_model_and_tokenizer(self, base_model, knowledge):
        """Load specific model and tokenizer"""
        self.knowledge_type = knowledge

        # Build model path
        if knowledge == "None":
            model_name = f"relevance_model {base_model}"
        else:
            vector_type = "Linear+RC" if knowledge == "RC" else "Linear+RP"
            model_name = f"relevance_model {base_model} ({vector_type})"

        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root/"models"/"fine-tuned"/f"{model_name}.pth"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Load model
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

        # Load tokenizer - assuming similar structure to your main code
        tokenizer_path = project_root/"models"/f"{base_model}"
        if not os.path.exists(tokenizer_path):
            # Fallback: try loading from transformers hub
            tokenizer_path = base_model.replace(" - ", "-").lower()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizer: {str(e)}")

    def _prepare_prediction_data(self, data):
        """Prepare data in the format expected by the dataset"""
        # Create DataFrame with expected format (Review column + dummy label)
        prediction_data = pd.DataFrame()
        # Add dummy column for labels (required by dataset)
        prediction_data["Review"] = data["Review"]
        prediction_data['Relevant'] = 0  # Dummy value, not used in prediction

        return prediction_data

    def _create_dataset(self, data):
        """Create appropriate dataset based on knowledge type"""
        if self.knowledge_type == "None":
            return CommentDataset(data, self.tokenizer, self.max_token_len)
        else:
            feature_type = "RELEVANT_COUNT" if self.knowledge_type == "RC" else "RELEVANT_POSITION"
            return CommentDatasetFV(data, self.tokenizer, self.max_token_len, feature_type)

    def _create_dataloader(self, dataset):
        """Create appropriate dataloader"""
        if self.knowledge_type == "None":
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0  # Use 0 to avoid issues in some environments
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_function
            )

    def _predict(self, dataloader):
        """Make predictions using the loaded model"""
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Move batch to device
                    if self.knowledge_type == "None":
                        batch["input_ids"] = batch["input_ids"].to(self.device)
                        batch["attention_mask"] = batch["attention_mask"].to(self.device)
                        # Normal model
                        _, outputs = self.model(
                            batch["input_ids"],
                            batch["attention_mask"]
                        )
                    else:
                        batch["input_ids"] = batch["input_ids"].to(self.device)
                        batch["attention_mask"] = batch["attention_mask"].to(self.device)
                        batch["feature_vectors"] = batch["feature_vectors"].to(self.device)
                        # Model with feature vectors
                        _, outputs = self.model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            feature_vectors=batch["feature_vectors"]
                        )

                    # Convert to binary predictions (0 or 1)
                    binary_preds = (outputs > 0.5).int()
                    predictions.extend(binary_preds.cpu().numpy().flatten())

                except Exception as e:
                    raise RuntimeError(f"Error during prediction: {str(e)}")

        return predictions

    def _filter_results(self, original_dataframe, predictions):
        """Filter original DataFrame based on predictions"""
        if len(predictions) != len(original_dataframe):
            print(predictions)
            raise ValueError("Number of predictions doesn't match number of rows")

        # Create mask for relevant comments (class 1)
        relevant_mask = [pred == 1 for pred in predictions]

        # Filter original DataFrame
        filtered_df = original_dataframe[relevant_mask].copy()

        # Reset indices
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df

    def set_batch_size(self, batch_size):
        """Set batch size configuration"""
        self.batch_size = batch_size

    def set_max_token_length(self, max_length):
        """Set maximum token length configuration"""
        self.max_token_len = max_length


# Example usage
if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Create sample data
    sample_data = pd.DataFrame({
        'Review': [
            # informative
            "And I want swift key to save the words I type",
            "I think it could do with a bit more customization...",
            "Does not work with S-Pen on Galaxy Note 2",
            "I just need Chinese!!!",
            "Uninstalled until I can save it to the SD card, everything is great except for the amount of space it takes up",
            "It would be perfect if there was a delete key!",
            "Just wish it had more themes",
            "Please add a few themes frequently",
            # non-informative
            "Stock keyboard, just as good",
            "I first downloaded the free trial but once that ran out, I had to have the full version.",
            "It predicts not only words but sentences",
            "Not a big deal for me but it looks bad when someone's watching my phone",
            "The predictions are depressingly accurate",
            "The interface and special key layout are the best I've seen",
            "Easy to use, the more you use it the better it gets",
            "The best, excellent keyboard!!!"
        ]
    })

    # Create filtering module instance
    filter_module = FilteringModule()

    try:
        # Example with normal model (no feature vectors)

        print("\n=== Filtering with RC vector ===")
        # Example with relevant count vector
        filtered_rc = filter_module.filter_comments(
            sample_data,
            "BERTweet - base",
            "RC"
        )
        print(f"Relevant comments found: {len(filtered_rc)}")
        print(filtered_rc)

    except Exception as e:
        print(f"Error: {str(e)}")
