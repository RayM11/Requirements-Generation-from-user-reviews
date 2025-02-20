import pandas as pd
import torch
import warnings
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from Code.classes.CommentFilter import CommentFilter
from transformers import AutoTokenizer

from Code.classes.CommentDataModule import CommentDataModule


def metrics_average(metrics_list):

    metrics_sum = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "F1-score": 0
    }

    for dic in metrics_list:
        for metric in metrics_sum:
            metrics_sum[metric] += dic[metric]

    return {metric: sum_m / len(metrics_list) for metric, sum_m in metrics_sum.items()}


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data_path = "../data/facebook_labeled.csv"
    model = "../models/roberta - base"
    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = pd.read_csv(data_path)
    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)

    config = {
        'model': model,
        'batch_size': 16,
        'lr': 2e-5,
        'warmup': 0.2,
        'train_size': None,
        'weight_decay': 0.001,
        'max_token_len': 200,
        'n_epochs': 2
    }

    # datamodule
    data_module = CommentDataModule(train_data, test_data, tokenizer, batch_size=16, max_token_len=config['max_token_len'])
    data_module.setup()

    config["train_size"] = len(data_module.train_dataloader())

    # model init
    model = CommentFilter(config)

    # training
    trainer = pl.Trainer(max_epochs=config['n_epochs'], log_every_n_steps=5, enable_progress_bar=True)
    trainer.fit(model, data_module)

    # evaluation
    trainer.test(model, data_module)
    metrics = model.get_metrics()
    print(f"metrics: {metrics}")

    # saving the model
    save_path = f"../models/fine-tuned/roberta - facebook.pth"
    torch.save(model, save_path)

    # data_module = CrossCommentDataModule(1, 5, split_seed, dataset, tokenizer, batch_size=16)
    # data_module.setup()
    # model = torch.load(fine_tuned)
    # config["train_size"] = len(data_module.train_dataloader())
    # model = CommentClassifier(config)
##
    # trainer = pl.Trainer(max_epochs=config['n_epochs'], log_every_n_steps=5, enable_progress_bar=True)
    # trainer.fit(model, data_module)
##
    # save_path = f"C:/Users/rmaes/PycharmProjects/requirements_classifier/models/fine-tuned/facebook.pth"
    # torch.save(model, save_path)
##
    # trainer.test(model, data_module)
##
    # results = model.get_metrics()
    # print(results)
