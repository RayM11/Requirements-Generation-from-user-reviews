import pandas as pd
import torch
import warnings
import pytorch_lightning as pl
from classes.CommentFilter import CommentFilter
from transformers import AutoTokenizer, AutoModel
from classes.CrossCommentDataModule import CrossCommentDataModule
from classes.CommentFilterPondered import CommentFilterPondered
from utils import get_tokenized_glossary


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


def cross_validation_Relevance(model_name, data_name, n_folds, config, pondered_attention: bool):

    model_path = "../models/" + model_name
    data_path = "../data/" + data_name + ".csv"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    split_seed = 123
    results = []

    dataframe = pd.read_csv(data_path)

    for k in range(n_folds):
        # datamodule
        data_module = CrossCommentDataModule(k, n_folds, split_seed, dataframe, tokenizer, batch_size=16,
                                             max_token_len=config['max_token_len'])
        data_module.setup()

        config["train_size"] = len(data_module.train_dataloader())

        # model init
        if pondered_attention:
            base_model = AutoModel.from_pretrained(model_path)
            glossary = get_tokenized_glossary(base_model, tokenizer)
            relevance_model = CommentFilterPondered(config,glossary)
        else:
            relevance_model = CommentFilter(config)

        # training
        trainer = pl.Trainer(max_epochs=config['n_epochs'], log_every_n_steps=5, enable_progress_bar=True)
        trainer.fit(relevance_model, data_module)

        # evaluation
        trainer.test(relevance_model, data_module)
        metrics = relevance_model.get_metrics()

        results.append(metrics)
        print(f" {k+1}-Fold metrics: {metrics}")

        save_path = f"../models/fine-tuned/relevance_model {model_name}-{data_name}-K{k + 1}.pth"
        torch.save(model, save_path)

    print("Results :", results)
    print("Avg results: ", metrics_average(results))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    dataset = "facebook_labeled"
    model = "roberta - base"
    # fine_tuned = "../models/fine-tuned/comment_relevance_detector (facebook).pth"
    n_folds = 5

    config = {
        'model': "../models/" + model,
        'batch_size': 16,
        'lr': 2e-5,
        'warmup': 0.2,
        'train_size': None,
        'weight_decay': 0.001,
        'max_token_len': 200,
        'n_epochs': 2
    }

    cross_validation_Relevance(model, dataset, n_folds, config, True)

    #for k in range(n_folds):
    #    # datamodule
    #    data_module = CrossCommentDataModule(k, n_folds, split_seed, dataset, tokenizer, batch_size=16,
    #                                         max_token_len=config['max_token_len'])
    #    data_module.setup()
#
    #    config["train_size"] = len(data_module.train_dataloader())
#
    #    # model init
    #    model = CommentFilter(config)
#
    #    # training
    #    trainer = pl.Trainer(max_epochs=config['n_epochs'], log_every_n_steps=5, enable_progress_bar=True)
    #    trainer.fit(model, data_module)
#
    #    # evaluation
    #    trainer.test(model, data_module)
    #    metrics = model.get_metrics()
#
    #    results.append(metrics)
    #    print(f" {k+1}-Fold metrics: {metrics}")
#
    #    save_path = f"../models/fine-tuned/xlnt-templerun2-K{k+1}.pth"
    #    torch.save(model, save_path)
#
    #print("Results :", results)
    #print("Avg results: ", metrics_average(results))

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
