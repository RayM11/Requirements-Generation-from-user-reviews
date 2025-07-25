from Code.logic.commentFiltering.CommentDatasetFV import CommentDatasetFV
from Code.logic.commentFiltering.CrossCommentDataModule import CrossCommentDataModule
from sklearn.model_selection import KFold


class CrossCommentDataModuleFV(CrossCommentDataModule):

    def __init__(self, k_fold, n_folds, split_seed, full_dataset, tokenizer, batch_size, max_token_len, feature_vector_type):
        super().__init__(k_fold, n_folds, split_seed, full_dataset, tokenizer, batch_size, max_token_len)

        self.feature_vector_type = feature_vector_type

    def setup(self, stage=None):

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(self.full_dataset)]
        train_indexes, val_indexes = all_splits[self.k_fold]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_dataset = CommentDatasetFV(self.full_dataset.iloc[train_indexes], self.tokenizer,
                                              self.max_token_len, self.feature_vector_type)
        self.test_dataset = CommentDatasetFV(self.full_dataset.iloc[val_indexes], self.tokenizer,
                                             self.max_token_len, self.feature_vector_type)
