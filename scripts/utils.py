import pandas as pd
import torch


def load_glossary(path="../glossary/isoiecieee5652.csv"):

    df = pd.read_csv(path)

    glosario = df['Term'].tolist()
    return glosario


def get_tokenized_glossary(model, tokenizer, path="../glossary/isoiecieee5652.csv", use_embeddings=False):

    glossary = load_glossary(path)

    print("Procesando el glosario...")

    glossary_tokens = tokenizer(glossary, padding=True, truncation=True, return_tensors="pt")

    if use_embeddings:
        with torch.no_grad():
            glossary_tokens = model(**glossary_tokens).last_hidden_state.mean(dim=1)

    print("Glosario Procesado")

    return glossary_tokens


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

    data_path1 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/facebook_labeled.csv"
    data_path2 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/tapfish_labeled.csv"
    data_path3 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/swiftkey_labeled.csv"
    data_path4 = "C:/Users/rmaes/PycharmProjects/requirements_classifier/data/templerun2_labeled.csv"

    train_data1 = pd.read_csv(data_path1)
    train_data2 = pd.read_csv(data_path2)
    train_data3 = pd.read_csv(data_path3)
    train_data4 = pd.read_csv(data_path4)

    print("len facebook: ", len(train_data1), "    len tapfish: ", len(train_data2))
    print("len swiftkey: ", len(train_data3), "    len templerun2: ", len(train_data4))
    print("facebook relevant count: ", train_data1["Relevant"].sum())
    print("tapfish relevant count: ", train_data2["Relevant"].sum())
    print("swiftkey relevant count: ", train_data3["Relevant"].sum())
    print("templerun2 relevant count: ", train_data4["Relevant"].sum())
