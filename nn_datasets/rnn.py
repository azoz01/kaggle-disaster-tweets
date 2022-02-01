import torch
import pandas as pd
import os
import en_core_web_lg
from tqdm.auto import tqdm
from preprocessing.freq_filtering import tokenize_and_filter_irrevelant_words
from preprocessing.freq_filtering import filter_tokenized_by_sentence_len

tqdm.pandas()
nlp = en_core_web_lg.load()

class RnnDataset(torch.utils.data.Dataset):

    def __init__(self, text_type: str, min_word_freq: int = 3, min_sentence_len: int = 0):
        """
        Args:
            text_type (str): Type of text to be read. So far available:
                - clean_with_stopwords
                - clean
                - lemmatized_with_stopwords
                - lemmatized
                - stemmed
            min_word_freq (int): words with lower frequency in text than that are removed. Defaults to 3.
            min_sentence_len (int): sentences with lower length after word removal are dropped with corresponding entries in target. Defaults to 0.
        """
        df_text = pd.read_csv(os.path.join('prepared', f'{text_type}.csv'))
        y = pd.read_csv('prepared/target.csv')

        df_text_filered_tokenized = tokenize_and_filter_irrevelant_words(df_text, min_word_freq)
        df_text_filered_tokenized, y = filter_tokenized_by_sentence_len(df_text_filered_tokenized, y, min_sentence_len)

        self.embedded = self.__encode_and_pad(df_text_filered_tokenized)
        self.target = y.values.flatten().astype('float')

    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, idx):

        return self.embedded[idx], self.target[idx]