from transformers import AutoTokenizer
import torch


class System:

    _instance = None
    tokenizer = None
    model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        ptm_path = "../models/roBERTa - base"
        self.tokenizer = AutoTokenizer.from_pretrained(ptm_path)

        fine_tuned_path = "../models/fine-tuned/comment_relevance_detector (facebook).pth"
        self.model = torch.load(fine_tuned_path)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def predict_relevance_comment(self, comment):
        encoding = self.tokenizer.encode_plus(comment,
                                              add_special_tokens=True,
                                              max_length=512,
                                              return_token_type_ids=False,
                                              padding="max_length",
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        _, prediction = self.model(encoding["input_ids"], encoding["attention_mask"])
        prediction = prediction.flatten().item()

        return prediction
