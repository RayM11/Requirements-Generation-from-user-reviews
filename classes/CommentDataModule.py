import pytorch_lightning as pl
from torch.utils.data import DataLoader
from classes.CommentDataset import CommentDataset


class CommentDataModule(pl.LightningDataModule):

    def __init__(self, train_data, test_data, tokenizer, batch_size: int = 16, max_token_len: int = 200):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CommentDataset(self.train_data, self.tokenizer, self.max_token_len)
        self.test_dataset = CommentDataset(self.test_data, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2, shuffle=False)
