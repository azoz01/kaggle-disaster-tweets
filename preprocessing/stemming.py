from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd

p = PorterStemmer()


def __stem_sentence(sentence: str) -> str:
    """
    Stems all words in sentence
    Args:
        sentence (str):

    Returns:
        str: sentence with all stemmed words
    """
    return ' '.join(list(map(p.stem, word_tokenize(sentence))))


def stem(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stems all words is series
    Args:
        text_df (pd.DataFrame): dataframe with column 'text' containing strings

    Returns:
        pd.DataFrame: dataframe with 'text' column with stemmed words
    """
    return text_df.apply(lambda row: __stem_sentence(row[0]),
                         axis=1,
                         raw=True
                         ).rename('text')
