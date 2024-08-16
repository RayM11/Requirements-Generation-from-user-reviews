import torch
from transformers import AutoTokenizer


def predict_relevance_comment(tokenizer, trained_model, comment):

    encoding = tokenizer.encode_plus(comment,
                                     add_special_tokens=True,
                                     max_length=512,
                                     return_token_type_ids=False,
                                     padding="max_length",
                                     return_attention_mask=True,
                                     return_tensors='pt',
    )

    _, prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
    prediction = prediction.flatten().item()

    return prediction


def predict_one_by_one(tokenizer, trained_model, dataframe):

    predictions = []
    labels = []

    for index in range(500,  510):
        comment = str(dataframe.iloc[index].Review)
        prediction = predict_relevance_comment(tokenizer, trained_model, comment)
        label = int(dataframe.iloc[index].Relevant)

        predictions.append(prediction)
        labels.append(label)

    return predictions, labels


def predict_relevance_dataset(trained_model, dataset):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    predictions = []
    labels = []

    for item in dataset:
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    return predictions, labels


if __name__ == '__main__':

    predict_path = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/tapfish_labeled (clean).csv"
    ptm_path = "C:/Users/rmaes/PycharmProjects/requirements_classifier/models/roBERTa - base"
    fine_tuned_path = "C:/Users/rmaes/PycharmProjects/requirements_classifier/models/fine-tuned/comment_relevance_detector (facebook).pth"
    tokenizerM = AutoTokenizer.from_pretrained(ptm_path)
    model = torch.load(fine_tuned_path)

    print(predict_relevance_comment(tokenizerM, model, "i have to go to the web for that and i cant organize my photos in certain albums only in the web!"))
