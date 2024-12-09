from classes.CommentFilter import CommentFilter
from torch import nn
import torch


class CommentFilterFV(CommentFilter):

    def __init__(self, config: dict):
        super().__init__(config)

        self.vector_size = self.pretrained_model.config.hidden_size + config['max_token_len']

        self.linear_classifier = torch.nn.Linear(self.vector_size, 1)
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight)

        self.use_mlp = config['use_mlp']

        self.multilayer_perceptron = nn.Sequential(
            nn.Linear(self.vector_size, self.vector_size),
            nn.ReLU(),
            nn.Linear(self.vector_size, self.vector_size),
            nn.ReLU(),
            self.linear_classifier
        )

    def forward(self, input_ids, attention_mask, labels=None, feature_vectors=None):

        # roberta layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state.mean(dim=1)

        output_with_domain_information = torch.cat((output, feature_vectors), dim=1)

        if self.use_mlp:
            logits = self.multilayer_perceptron(output_with_domain_information)
        else:
            logits = self.linear_classifier(output_with_domain_information)

        # logits = self.classifier(output_with_domain_information)
        logits = self.sigmoid(logits)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        feature_vectors = batch["feature_vector"]
        loss, outputs = self.forward(input_ids, attention_mask, labels, feature_vectors)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        feature_vectors = batch["feature_vector"]
        loss, outputs = self(input_ids, attention_mask, labels, feature_vectors)

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
        feature_vectors = batch["feature_vector"]
        loss, outputs = self(input_ids, attention_mask, labels, feature_vectors)

        metrics = {
            "train_loss": loss,
            "accuracy": self.accuracy(outputs, labels),
        }

        self.predictions.append(outputs)
        self.references.append(labels)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

