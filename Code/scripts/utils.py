import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def load_glossary(path="../glossary/isoiecieee5652.csv"):

    glossary = pd.read_csv(path)
    glossary['Term'] = glossary['Term'].str.strip()

    # glosario = df['Term'].tolist()

    return glossary


def count_relevant_terms(comment, glossary):
    count = 0

    for term in glossary['Term']:
        if term in comment:
            # print("found ", term)
            count += 1

    return count


def build_count_feature_vector(size, comment, glossary, full=True):

    feature_vector = torch.zeros(size)
    relevant_count = count_relevant_terms(comment, glossary)

    if full:
        feature_vector[:] = relevant_count
    else:
        feature_vector[0] = relevant_count

    return feature_vector


def build_position_feature_vector(size, comment, glossary, tokenizer):

    feature_vector = torch.zeros(size)
    tokenized_comment = tokenizer.encode(comment)
    # tokenized_glossary = get_tokenized_glossary(tokenizer, glossary)

    for term in glossary["Term"]:

        term_without_space = tokenizer.encode(term, add_special_tokens=False)
        term_with_space = tokenizer.encode(" " + term, add_special_tokens=False)

        apply_positional_match(feature_vector, tokenized_comment, term_without_space)
        apply_positional_match(feature_vector, tokenized_comment, term_with_space)

    return feature_vector


def apply_positional_match(feature_vector, tokenized_comment, tokenized_term):

    for i in range(len(tokenized_comment) - len(tokenized_term) + 1):

        # detokenized_fragment = tokenizer.decode(tokenized_comment[i:i + len(tokenized_term)])
        # detokenized_fragment = detokenized_fragment.strip()
        # detokenized_term = tokenizer.decode(tokenized_term)
        # detokenized_term = detokenized_term.strip()

        if tokenized_comment[i:i + len(tokenized_term)] == tokenized_term:
            # print("Found: ", tokenized_term)
            for j in range(len(tokenized_term)):
                feature_vector[i + j] = 1


def collate_function(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    feature_vectors = torch.tensor([item["feature_vector"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "feature_vectors": feature_vectors
    }


def get_tokenized_glossary(tokenizer, glossary):

    token_list = []

    # glossary["Term"] = glossary["Term"].apply(lambda text: tokenizer.encode(text))

    for term in glossary["Term"]:
        new_token = tokenizer.encode(term, add_special_tokens=False)
        # if len(new_token) == 1:
        token_list.append(new_token)

    tokenized_glossary = pd.DataFrame()
    tokenized_glossary["Term"] = token_list

    # glossary_tokens.extend(tokenized_term['input_ids'][1:-1])
    # print("\nglossary_tokens: ", glossary_tokens.__sizeof__())
    # glossary_tokens = list(set(glossary_tokens))
    # print("\nglossary_tokens: ", glossary_tokens.__sizeof__())

    # if use_embeddings:
    #    with torch.no_grad():
    #        glossary_tokens = model(**glossary_tokens).last_hidden_state.mean(dim=1)

    return tokenized_glossary


# def data_from_balanced(path: str):
#     data = pd.read_csv(path)
#
#     train_data, test_data = train_test_split(data, test_size=0.3, stratify=data["Relevant"])
#
#     return train_data, test_data
#
#
# def data_from_unbalanced(path: str):
#     data = pd.read_csv(path)
#
#     majority = data[data["Relevant"] == 0]
#     minority = data[data["Relevant"] == 1]
#     minority_train = minority.sample(frac=0.7, replace=True)
#     majority_train = majority.sample(len(minority_train), replace=True)
#     train_data = pd.concat([minority_train, majority_train])
#
#     test_data = data[~data.index.isin(train_data.index)]
#
#     return train_data, test_data
#
#
# def force_balance(path: str):
#     data = pd.read_csv(path)
#
#     majority = data[data["Relevant"] == 0]
#     minority = data[data["Relevant"] == 1]
#
#     majority = majority.sample(len(minority), replace=True)
#
#     return pd.concat([minority, majority], ignore_index=True)


if __name__ == '__main__':

    # data_path1 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/facebook_labeled.csv"
    # data_path2 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/tapfish_labeled.csv"
    # data_path3 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/swiftkey_labeled.csv"
    # data_path4 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/templerun2_labeled.csv"
#
    # train_data1 = pd.read_csv(data_path1)
    # train_data2 = pd.read_csv(data_path2)
    # train_data3 = pd.read_csv(data_path3)
    # train_data4 = pd.read_csv(data_path4)
#
    # print("len facebook: ", len(train_data1), "    len tapfish: ", len(train_data2))
    # print("len swiftkey: ", len(train_data3), "    len templerun2: ", len(train_data4))
    # print("facebook relevant count: ", train_data1["Relevant"].sum())
    # print("tapfish relevant count: ", train_data2["Relevant"].sum())
    # print("swiftkey relevant count: ", train_data3["Relevant"].sum())
    # print("templerun2 relevant count: ", train_data4["Relevant"].sum())

    glossary = load_glossary()

    comment = "i wish i could give more stars!!!!!!!"
    # comment2 = "accident"
    # comment3 = "The access"

    tokenizer = AutoTokenizer.from_pretrained("../models/BERTweet - base")
    # model = AutoModel.from_pretrained("../models/roBERTa - base")
    class_model = torch.load("../models/fine-tuned/relevance_model BERTweet - base (Linear+RC) - swiftkey - K4.pth")

    relevant_count = build_count_feature_vector(130, comment, glossary)
    relevant_position = build_position_feature_vector(130, comment, glossary, tokenizer)

    encoding = tokenizer.encode_plus(comment,
                                     add_special_tokens=True,
                                     max_length=130,
                                     return_token_type_ids=False,
                                     padding="max_length",
                                     return_attention_mask=True,
                                     return_tensors='pt'
                                     )

    feature_vector = relevant_count[np.newaxis, :]

    _, prediction = class_model(encoding["input_ids"], encoding["attention_mask"], feature_vectors=feature_vector)
    prediction = prediction.flatten().item()

    print("Comment: ", comment)
    print("Relevant_Count: ", relevant_count)

    print("Comment: ", comment)
    print("Relevant_Position: ", relevant_position)

    # inputs = tokenizer(comment, return_attention_mask=False)["input_ids"]
    # inputs2 = tokenizer(comment2, return_attention_mask=False)["input_ids"]
    # inputs3 = tokenizer(comment3, return_attention_mask=False)["input_ids"]
    # inputs4 = tokenizer("access", return_attention_mask=False, add_special_tokens=False)["input_ids"]
    #
    # print(inputs)
    # print(tokenizer.decode(inputs[10]))
    # print(inputs2)
    # print(tokenizer.decode(inputs2[1:3]))
    # print(inputs3)
    # print(inputs4)
