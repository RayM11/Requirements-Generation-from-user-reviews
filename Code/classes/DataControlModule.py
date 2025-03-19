import pandas as pd


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


def load_comments(csv_path):
    """Load comments from a CSV file.

    Args:
        csv_path (str): Path to the CSV file with comments

    Returns:
        pandas.DataFrame: DataFrame with loaded comments
    """
    # Load CSV without header and with a single column
    df = pd.read_csv(csv_path, header=None, names=['Review'])
    return df


def save_results(df, output_path):
    """Save the results to a CSV file.

    Args:
        df (pandas.DataFrame): DataFrame with comments and classifications
        output_path (str): Path to save the CSV file

    Returns:
        None
    """
    df.to_csv(output_path, index=False)
    return df

