from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd

from preprocessing.embedding import encode_df_tokenized
from preprocessing.freq_filtering import (filter_tokenized_by_sentence_len,
                                          tokenize_and_filter_irrevelant_words)


def encode_df_text(df_text: pd.DataFrame, target: pd.DataFrame, min_word_freq: int, 
                  min_sentence_len: int, embedding: Callable[[str], int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenizes sentences, removes words with lower than min_word_freq,
    removes entries from df_text and target when lenght of entry is df_text is lower than min_sentence_len,
    and encodes using embedding mapping 

    Args:
        df_text (pd.DataFrame): dataframe with 'text' column containing strings
        target (pd.DataFrame): target variable from dataset
        min_word_freq (int): minimum frequency of word in dataset
        min_sentence_len (int): minimum length of sentence
        embedding (Callable[[str], int]): mapping from string to vector with fixed length

    Returns:
        Tuple[np.ndarray, np.ndarray]: first are embedded sentences from df_text, second target (after filtering)
    """
    tokenized = tokenize_and_filter_irrevelant_words(df_text, min_word_freq)
    tokenized, target = filter_tokenized_by_sentence_len(tokenized, target, min_sentence_len)
    return encode_df_tokenized(tokenized, embedding), target.values.squeeze(1)
