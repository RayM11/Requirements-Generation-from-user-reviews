import torch
import math
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
from torchmetrics.classification import ConfusionMatrix


class CommentFilter(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model'], return_dict=True)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCELoss()
        self.num_classes = 2

        self.predictions = []
        self.references = []

        # metrics
        self.accuracy = tm.Accuracy(task="binary")
        self.precision = tm.Precision(task="binary")
        self.recall = tm.Recall(task="binary")
        self.f1_score = tm.F1Score(task="binary")

    def forward(self, input_ids, attention_mask, labels=None):
        # roberta layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        # final logits
        output = self.classifier(output.last_hidden_state.mean(dim=1))
        logits = self.sigmoid(output)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)

        metrics = {
            "val_loss": loss,
            "accuracy": self.accuracy(outputs, labels),
            "precision": self.precision(outputs, labels),
            "recall": self.recall(outputs, labels),
            "F1-score": self.f1_score(outputs, labels)
        }

        self.predictions.append(outputs)
        self.references.append(labels)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)

        metrics = {
            "train_loss": loss,
            "accuracy": self.accuracy(outputs, labels),
        }

        self.predictions.append(outputs)
        self.references.append(labels)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        return outputs

    def on_test_start(self):
        self.predictions.clear()
        self.references.clear()

    def on_test_end(self):
        predictions = (torch.concat(self.predictions) > 0.5).int()
        labels = torch.concat(self.references)
        confusion_mat = ConfusionMatrix(task="binary", num_classes=2)
        print("Cunfusion Matrix: \n", confusion_mat(predictions, labels))

    def get_metrics(self):

        predictions = (torch.concat(self.predictions) > 0.5).int()
        labels = torch.concat(self.references)
        confusion_mat = ConfusionMatrix(task="binary", num_classes=2)
        conf_result = confusion_mat(predictions, labels)

        tp = float(conf_result[0, 0].item())
        fp = float(conf_result[0, 1].item())
        fn = float(conf_result[1, 0].item())
        tn = float(conf_result[1, 1].item())

        print(confusion_mat, tp, fp, fn, tn)

        return {
            "accuracy": (tp+tn)/(tp+fp+fn+tn),
            "precision": tp/(tp+fp),
            "recall": tp/(tp+fn),
            "F1-score": 2*((tp/(tp+fp)*tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn)))
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
