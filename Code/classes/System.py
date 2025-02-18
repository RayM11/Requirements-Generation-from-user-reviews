import numpy as np
from transformers import AutoTokenizer
import torch

from Code.scripts.utils import build_position_feature_vector, load_glossary

class System:

    _instance = None
    tokenizer = None
    model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        ptm_path = "models/roBERTa - base"
        self.tokenizer = AutoTokenizer.from_pretrained(ptm_path)

        fine_tuned_path = "models/fine-tuned/relevance_model roBERTa - base (Linear+RP) -facebook-K3.pth"
        self.model = torch.load(fine_tuned_path)

        glossary_path = "glossary/isoiecieee5652.csv"
        self.glossary = load_glossary(glossary_path)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_glossary(self):
        return self.glossary

    def predict_relevance_comment(self, comment):
        encoding = self.tokenizer.encode_plus(comment,
                                              add_special_tokens=True,
                                              max_length=200,
                                              return_token_type_ids=False,
                                              padding="max_length",
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )
        feature_vector = build_position_feature_vector(200, comment, self.glossary, self.tokenizer)
        feature_vector = feature_vector[np.newaxis, :]

        _, prediction = self.model(encoding["input_ids"], encoding["attention_mask"], feature_vectors=feature_vector)
        prediction = prediction.flatten().item()

        return prediction
