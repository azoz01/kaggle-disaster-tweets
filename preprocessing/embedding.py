from collections import Counter
from functools import partial
from typing import Callable, DefaultDict, Dict, Iterable, List

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def encode_df_tokenized(df_tokenized: pd.DataFrame, embedding: Callable[[str], np.ndarray]) -> np.ndarray:
    """
    Args:
        df_tokenized (pd.DataFrame): dataframe containing column 'text' with lists of words
        embedding (Callable[[str], np.ndarray]): mapping from string to vector with fixed length

    Returns:
        np.ndarray: array where rows are encoded rows od df_tokenized
    """
    if embedding in [one_hot, word_count_encoding]:
        id_dict = __get_id_dict_of_words(df_tokenized)
        embedding = partial(embedding, id_dict = id_dict)
        return np.array(list(map(embedding, df_tokenized.values)))
    else:
        return np.apply_along_axis(embedding, 1, df_tokenized.values)


def __get_id_dict_of_words(df_tokenized: pd.DataFrame) -> Dict[str, int]:
    """
    Creates dictionary which keays are words from df_tokenized and values are unique ids for each word

    Args:
        df_tokenized (pd.DataFrame): dataframe containing column 'text' with lists of words

    Returns:
        Dict[str, int]: dictionary where words are keys and values are unique ids
    """
    id_generator = __increment_id_generator()
    words = word_tokenize(' '.join(map(lambda s: ' '.join(s), df_tokenized)))
    id_dict = DefaultDict(lambda: next(id_generator))
    for word in words:
        id_dict[word]
    return dict(id_dict)


def __increment_id_generator() -> Iterable[int]:
    """
    Generator, which generates each new id by incrementation

    Returns:
        Iterable[int]: Generator of ids

    Yields:
        Iterator[Iterable[int]]: unique id
    """
    id = 0
    while 1:
        yield id
        id += 1


def one_hot(words: List[str], id_dict: Dict[str, int]) -> np.ndarray:
    """
    Args:
        words (List[str]): list of words
        id_dict (Dict[str, int]): dictionary containing words in keys and ids in values

    Returns:
        np.ndarray: vector where v_i = 1 when word with id i, 0 otherwise
    """
    ids = [id_dict[word] for word in words]
    vector = np.zeros(len(id_dict))
    vector[ids] = 1
    return vector


def word_count_encoding(words: List[str], id_dict: Dict[str, int]) -> np.ndarray:
    """
    Args:
        words (List[str]): list of words
        id_dict (Dict[str, int]): dictionary containing words in keys and ids in values

    Returns:
        np.ndarray: vector where i'th element contains number of occurences words with id i
    """
    ids = [id_dict[word] for word in words]
    counter = Counter(ids)
    vector = np.zeros((len(id_dict)))
    vector[ids] = [counter[id] for id in ids]
    return vector
