from classes.CommentFilter import CommentFilter
import torch
from scripts.utils import load_glossary


class CommentFilterPondered(CommentFilter):

    def __init__(self, config: dict, glossary_tokens):
        super().__init__(config)

        self.glossary_tokens = glossary_tokens

    def forward(self, input_ids, attention_mask = None, labels=None):

        # Crear la máscara de atención si no se proporciona
        if attention_mask is None:
            attention_mask = (input_ids != 0).int()

        print("attention Mask: ", attention_mask)

        # Ajustar la máscara de atención para los tokens del glosario
        for i, row in enumerate(input_ids):
            for j, token_id in enumerate(row):
                if token_id in self.glossary_tokens:
                    attention_mask[i, j] = 2

        if 2 in attention_mask:
            print("Hay 2")

        print("attention Mask: ", attention_mask)

        # roberta layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state


        # Calcular similitudes de coseno entre los embeddings de los tokens y los del glosario
        # similarity_scores = torch.einsum('bld,gd->blg', sequence_output, self.glossary_embeddings)
        # max_similarities, _ = torch.max(similarity_scores, dim=-1)
#
        # print('Max Similarities: \n', max_similarities)
        # print("Max sim: ", torch.max(max_similarities))
        # print("Min sim: ", torch.min(max_similarities))
#
        # # Incrementar las puntuaciones de atención en función de las similitudes
        # attention_weights = torch.nn.functional.softmax(max_similarities, dim=-1)
        # attention_weights = torch.where(attention_weights > 160, 1, 2)
#
        # print('Attention Weights:\n', attention_weights)
#
        # # Aplicar las nuevas puntuaciones de atención al output del modelo base
        # weighted_output = sequence_output * attention_weights.unsqueeze(-1)

        #print('Sequence Output:\n', sequence_output)
        #print('Weighted Output:\n', weighted_output)

        # final logits
        # output = self.classifier(output.last_hidden_state.mean(dim=1))
        logits = self.classifier(output.mean(dim=1))
        logits = self.sigmoid(logits)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

