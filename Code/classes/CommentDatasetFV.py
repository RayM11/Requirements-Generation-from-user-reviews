import warnings

import pandas as pd
import torch
from Code.classes.CommentDataset import CommentDataset
from Code.scripts.utils import load_glossary, build_count_feature_vector, build_position_feature_vector

max_len_btweet = 500


class CommentDatasetFV (CommentDataset):

    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len, feature_vector_type="RELEVANT_COUNT"):
        super().__init__(data, tokenizer, max_token_len)
        self.glossary = load_glossary()
        self.feature_vector_type = feature_vector_type

    def __getitem__(self, index):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        item = self.data.iloc[index]
        comment = str(item.Review)
        comment = comment[:max_len_btweet] if len(comment) > max_len_btweet else comment
        print("Comment ", index, ": ", comment)
        label = torch.LongTensor(self.data.iloc[index, 1:])
        encoding = self.tokenizer.encode_plus(
                                comment,
                                add_special_tokens=True,
                                max_length=self.max_token_len,
                                return_token_type_ids=False,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
        )
        feature_vector = None
        if self.feature_vector_type == "RELEVANT_COUNT":
            feature_vector = build_count_feature_vector(self.max_token_len, comment, self.glossary, full=True)

        elif self.feature_vector_type == "RELEVANT_POSITION":
            feature_vector = build_position_feature_vector(self.max_token_len, comment, self.glossary, self.tokenizer)

        else:
            raise ValueError(f"ERROR: Invalid feature vector type: {self.feature_vector_type}")

        if len(encoding["input_ids"].flatten()) != self.max_token_len:
            print("Bad length, expected ", self.max_token_len, ", got ", len(encoding["input_ids"].flatten().len()))

        return {'input_ids': encoding["input_ids"].flatten(),
                'attention_mask': encoding["attention_mask"].flatten(),
                'label': label,
                'feature_vector': feature_vector}
