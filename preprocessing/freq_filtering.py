from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def tokenize_and_filter_irrevelant_words(df_text: pd.DataFrame, min_word_freq: int) -> pd.DataFrame:
        """
        Removes from datasets words which frequencies are lower than min_count

        Args:
            df_text (pd.DataFrame): dataframe with 'text' column containing strings
            min_word_freq (int): minimum frequency of word in dataset

        Returns:
            pd.DataFrame: dataframe containing tokenized sentences
        """
        all_word_counter = dict(Counter(word_tokenize(' '.join(df_text['text']))))
        dict_array = np.array(list(all_word_counter.items()))
        significant = dict_array[dict_array[:, 1].astype('uint32') > min_word_freq]
        frequency_filter = lambda row: [word for word in word_tokenize(row[0]) if word in significant[:, 0]]
        df_text_filered_tokenized = df_text.apply(frequency_filter, axis = 1)
        return df_text_filered_tokenized


def filter_tokenized_by_sentence_len(tokenized_df: pd.DataFrame, target: pd.DataFrame, min_sentence_len: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes entries from datasets where length of sentence is lower than min_len

        Args:
            tokenized_df (pd.DataFrame): dataframe with 'text' column containing tokenized sentences
            target (pd.DataFrame): target variable from dataset
            min_sentence_len (int): minimum length of sentence

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: dataframes with removed entries
        """
        len_mask = np.array(list(map(len, tokenized_df.values))) >= min_sentence_len
        return tokenized_df.loc[len_mask].reset_index(drop = True), target.loc[len_mask].reset_index(drop = True)
