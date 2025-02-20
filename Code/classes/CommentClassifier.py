import torch
import math
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup


class CommentFilter(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model'], return_dict=True)
        self.linear_classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, config['num_classes'])
        self.softmax = nn.Softmax(dim=1)
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = config['num_classes']

        self.predictions = []
        self.references = []

        # metrics
        self.confusion_matrix = tm.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.precision = tm.Precision(task="multiclass", num_classes=self.num_classes, average='macro')
        self.recall = tm.Recall(task="multiclass", num_classes=self.num_classes, average='macro')
        self.f1_score = tm.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')

    def forward(self, input_ids, attention_mask, labels=None):
        # roberta layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        # final logits
        logits = self.linear_classifier(output.last_hidden_state.mean(dim=1))
        # logits = self.softmax(output)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, logits = self.forward(input_ids, attention_mask, labels)

        preds = torch.argmax(logits, dim=1)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "logits": logits, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, dim=1)

        metrics = {
            "val_loss": loss,
            "accuracy": self.accuracy(preds, labels),
            "precision": self.precision(preds, labels),
            "recall": self.recall(preds, labels),
            "F1-score": self.f1_score(preds, labels)
        }

        self.predictions.append(preds)
        self.references.append(labels)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": preds, "labels": labels}

    def test_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, logits = self(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, dim=1)

        metrics = {
            "train_loss": loss,
            "accuracy": self.accuracy(preds, labels),
        }

        self.predictions.append(preds)
        self.references.append(labels)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "preds": preds, "labels": labels}

    def predict_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        return outputs

    def on_test_start(self):
        self.predictions.clear()
        self.references.clear()

    #def on_test_end(self):
    #    predictions = torch.concat(self.predictions)
    #    labels = torch.concat(self.references)
    #    confusion_mat = self.confusion_matrix(predictions, labels)
#
    #    print("Cunfusion Matrix: \n", confusion_mat)
    #    print(f"Class-wise Precision: {self.precision(predictions, labels, average=None)}")
    #    print(f"Class-wise Recall: {self.recall(predictions, labels, average=None)}")

    def get_metrics(self):

        predictions = torch.concat(self.predictions)
        labels = torch.concat(self.references)

        metrics = {
            "accuracy": self.accuracy(predictions, labels),
            "precision": self.precision(predictions, labels),
            "recall": self.recall(predictions, labels),
            "F1-score": self.f1_score(predictions, labels)
        }

        #tp = float(confusion_mat[0, 0].item())
        #fp = float(confusion_mat[0, 1].item())
        #fn = float(confusion_mat[1, 0].item())
        #tn = float(confusion_mat[1, 1].item())
#
        #print(confusion_mat, tp, fp, fn, tn)
#
        #return {
        #    "accuracy": (tp+tn)/(tp+fp+fn+tn),
        #    "precision": tp/(tp+fp),
        #    "recall": tp/(tp+fn),
        #    "F1-score": 2*((tp/(tp+fp)*tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn)))
        #}

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
