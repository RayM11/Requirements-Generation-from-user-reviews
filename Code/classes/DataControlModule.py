import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def load_glossary(path="../glossary/isoiecieee5652.csv"):

    glossary = pd.read_csv(path)
    glossary['Term'] = glossary['Term'].str.strip()

    # glosario = df['Term'].tolist()

    return glossary


def get_tokenized_glossary(tokenizer, glossary):

    token_list = []

    # glossary["Term"] = glossary["Term"].apply(lambda text: tokenizer.encode(text))

    for term in glossary["Term"]:
        new_token = tokenizer.encode(term, add_special_tokens=False)
        # if len(new_token) == 1:
        token_list.append(new_token)

    tokenized_glossary = pd.DataFrame()
    tokenized_glossary["Term"] = token_list

    return tokenized_glossary

