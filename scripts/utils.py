import pandas as pd
import torch


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
    tokenized_glossary = get_tokenized_glossary(tokenizer, glossary)

    for i, token in enumerate(tokenized_comment):
        if token < 4 and token in tokenized_glossary:
            feature_vector[i] = 1

    return feature_vector


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
        token_list.append(tokenizer.encode(term, add_special_tokens=False))

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

    comment = "I don´t find the access method, accident"

    relevant_count = count_relevant_terms(comment, glossary)

    feature_vector = build_feature_vector(200, relevant_count)

    print("Comment: ", comment)
    print("Relevant_count: ", relevant_count)
    print("Feature_vector: ", feature_vector)
