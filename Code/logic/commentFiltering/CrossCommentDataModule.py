import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Code.logic.commentFiltering.CommentDataset import CommentDataset
from sklearn.model_selection import KFold


class CrossCommentDataModule(pl.LightningDataModule):
    def __init__(self, k_fold, n_folds, split_seed, full_dataset, tokenizer, batch_size, max_token_len):
        super().__init__()
        self.full_dataset = full_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

        # actual fold number
        self.k_fold = k_fold
        # number of folds
        self.n_folds = n_folds
        # seed to control the randomness of fold splitting
        self.split_seed = split_seed

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(self.full_dataset)]
        train_indexes, val_indexes = all_splits[self.k_fold]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_dataset = CommentDataset(self.full_dataset.iloc[train_indexes], self.tokenizer, self.max_token_len)
        self.test_dataset = CommentDataset(self.full_dataset.iloc[val_indexes], self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                          persistent_workers=True,)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          persistent_workers=True,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          persistent_workers=True,)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False,
                          persistent_workers=True,)

