import re
from nltk.tokenize import word_tokenize
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    # Removing special characters
    text = re.sub(r'[^\w\s.,:]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text in words
    tokens = word_tokenize(text)

    # Correction of spelling errors
    spell = Speller()
    tokens = [spell(token) for token in tokens]

    # Removal of stopwords
    # if remove_stopwords:
        # stop_words = set(stopwords.words('english'))
        # Exclude specific words from the stopwords list
        # custom_stopwords = ("", "", "", "")
        # stop_words -= set(custom_stopwords)

        # tokens = [token for token in tokens if token not in stop_words]

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Joining tokenized words into a single text
    cleaned_text = ' '.join(tokens)

    return cleaned_text
