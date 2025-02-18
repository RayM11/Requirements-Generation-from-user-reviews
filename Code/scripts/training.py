import pandas as pd
import torch
import warnings
import pytorch_lightning as pl
from Code.classes.CommentFilter import CommentFilter
from Code.classes.CommentDataModule import CommentDataModule
from Code.classes.CommentDataset import CommentDataset
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    train_path = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/full_train_data.csv"
    test_path = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/facebook_test.csv"
    model = "C:/Users/rmaes/PycharmProjects/requirements_classifier/models/roBERTa - base"
    attributes = ["Relevant"]
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)
    # dataset = CommentDataset(train_dataframe, tokenizer)
    # dataset_eval = CommentDataset(test_dataframe, tokenizer)

    # datamodule
    data_module = CommentDataModule(train_dataframe, test_dataframe, tokenizer, batch_size=16, max_token_len=200)
    data_module.setup()

    config = {
        'model': 'C:/Users/rmaes/PycharmProjects/requirements_classifier/models/roBERTa - base',
        'batch_size': 32,
        'lr': 2e-5,
        'warmup': 0.2,
        'train_size': len(data_module.train_dataloader()),
        'weight_decay': 0.001,
        'n_epochs': 4
    }

    # model
    model = CommentFilter(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath="checkpoints",
      filename="best-checkpoint",
      save_top_k=1,
      verbose=True,
      monitor="val_loss",
      mode="min"
    )

    # trainer and fit
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         max_epochs=config['n_epochs'],
                         log_every_n_steps=5,
                         enable_progress_bar=True)
    trainer.fit(model, data_module)

    model_save_path = 'C:/Users/rmaes/PycharmProjects/requirements_classifier/models/comment_relevance_detector.pth'
    torch.save(model, model_save_path)
