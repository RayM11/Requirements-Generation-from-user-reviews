from pathlib import Path

import pandas as pd
import torch
import warnings
import pytorch_lightning as pl
from Code.logic.commentFiltering.CommentFilter import CommentFilter
from transformers import AutoTokenizer
from Code.logic.commentFiltering.CrossCommentDataModule import CrossCommentDataModule
from Code.logic.commentFiltering.CommentFilterFV import CommentFilterFV
from Code.logic.commentFiltering.CrossCommentDataModuleFV import CrossCommentDataModuleFV


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


def cross_validation_relevance(model_name, data_name, n_folds, config, mode="base"):

    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / "models" / model_name
    data_path = project_root / "data" / "datasets" / data_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    split_seed = 123
    results = []

    dataframe = pd.read_csv(data_path)

    for k in range(n_folds-4):

        data_module = None

        # model init
        if mode == "base":
            data_module = CrossCommentDataModule(k, n_folds, split_seed, dataframe, tokenizer,
                                                 batch_size=config['batch_size'], max_token_len=config['max_token_len'])
            relevance_model = CommentFilter(config)

        elif mode == "FV":
            data_module = CrossCommentDataModuleFV(k, n_folds, split_seed, dataframe, tokenizer,
                                                   batch_size=config['batch_size'], max_token_len=config['max_token_len'],
                                                   feature_vector_type=config['FV_type'])
            relevance_model = CommentFilterFV(config)
        else:
            raise ValueError(f"ERROR: Invalid relevance mode: {mode}")

        data_module.setup()

        config["train_size"] = len(data_module.train_dataloader())

        # training
        trainer = pl.Trainer(max_epochs=config['n_epochs'], log_every_n_steps=5, enable_progress_bar=True)
        trainer.fit(relevance_model, data_module)

        # evaluation
        trainer.test(relevance_model, data_module)
        metrics = relevance_model.get_metrics()

        results.append(metrics)
        print(f" {k+1}-Fold metrics: {metrics}")

        save_path = project_root / "models" / "fine-tuned" / f"relevance_model {model_name}-{data_name}-K{k + 1}.pth"
        torch.save(relevance_model, save_path)

    print("Results :", results)
    print("Avg results: ", metrics_average(results))


if __name__ == '__main__':
    warnings.simplefilter(action="ignore", category=FutureWarning)

    dataset = "swiftkey_labeled.csv"
    # model = "albert v2 - base"
    model = "BERTweet - base"
    # fine_tuned = "../models/fine-tuned/comment_relevance_detector (facebook).pth"
    n_folds = 5

    project_root = Path(__file__).resolve().parent.parent.parent

    config = {
        'model': project_root / "models" / model,
        'batch_size': 16,
        'lr': 2e-5,
        'warmup': 0.2,
        'train_size': None,
        'weight_decay': 0.001,
        'max_token_len': 130,
        'n_epochs': 2,

        'FV_type': 'RELEVANT_COUNT',    # RELEVANT_COUNT or RELEVANT_POSITION
        'use_mlp': False,
        'mlp_dimension': 100,
        "nlp_dimension_2": 23
    }

    cross_validation_relevance(model, dataset, n_folds, config, "FV")
