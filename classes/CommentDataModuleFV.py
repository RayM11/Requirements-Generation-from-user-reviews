from torch.utils.data import DataLoader
from classes.CommentDatasetFV import CommentDatasetFV
from classes.CommentDataModule import CommentDataModule
from scripts.utils import collate_function


class CommentDataModuleFV(CommentDataModule):

    def __init__(self, train_data, test_data, tokenizer, batch_size, max_token_len):
        super().__init__(train_data, test_data, tokenizer, batch_size, max_token_len)

    def setup(self, stage=None):
        self.train_dataset = CommentDatasetFV(self.train_data, self.tokenizer, self.max_token_len)
        self.test_dataset = CommentDatasetFV(self.test_data, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                          persistent_workers=True, collate_fn=collate_function)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          persistent_workers=True, collate_fn=collate_function)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          persistent_workers=True, collate_fn=collate_function)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          persistent_workers=True, collate_fn=collate_function)
