import os
from typing import Union

import pandas as pd
import torch

from transformers import AutoTokenizer


class BertDataset(torch.utils.data.Dataset):
    """
    Dataset compatible with BERT transformer from hugging face library
    """
    def __init__(self, text_type: str,
                 pretrained_tokenizer_name: str,
                 vector_len: int,
                 hugging_face: bool = False
                 ) -> None:
        """
        Args:
            text_type (str): Type of text to be read. So far available:
                - clean_with_stopwords
                - clean
                - lemmatized_with_stopwords
                - lemmatized
                - stemmed
            pretrained_tokenizer (str): name of tokenizer from hugging face
            vector_len (int): len of vector after tokenization and padding
            hugging_face (bool, optional): if True __get__ returns entries as
                dictionary compatible with transformers.Trainer,
                if False entries are in form (X, y). Defaults to False.

        Raises:
            ValueError: when given wrong text_type
        """
        try:
            texts_df = pd.read_csv(os.path.join('prepared',
                                                f'{text_type}.csv')
                                   )
        except FileNotFoundError:
            raise ValueError(f'Invalid text_type {text_type}')
        self.hugging_face = hugging_face

        self.texts = texts_df.values

        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name,
                                                  use_fast=False)
        self.texts = tokenizer(self.texts.squeeze(1).tolist(),
                               padding='max_length',
                               max_length=vector_len,
                               truncation=True)

        self.ids = self.texts['input_ids']
        self.token_type_ids = self.texts['token_type_ids']
        self.attention_mask = self.texts['attention_mask']

        self.labels = pd.read_csv('prepared/target.csv')
        self.labels = self.labels
        self.labels = torch.Tensor(self.labels.values).squeeze(1)

    def classes(self) -> torch.Tensor:
        return self.labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def get_batch_labels(self, idx: slice) -> torch.Tensor:
        return self.labels[idx]

    def get_batch_texts(self, idx) -> dict[str, list]:
        return {'input_ids': self.ids[idx],
                'token_type_ids': self.token_type_ids[idx],
                'attention_mask': self.attention_mask[idx]}

    def __getitem__(self,
                    idx: slice
                    ) -> Union[tuple[torch.Tensor, torch.Tensor],
                               dict[str, Union[torch.Tensor, list]]]:
        """
        If self.hugging_face == True returns entries as
        dictionary compatible with transformers.Trainer,
        if False entries are in form (X, y).

        Args:
            idx (slice): indices

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor],
                  dict[str, Union[torch.Tensor, list]]]:
                    entries with idx indices
        """
        if self.hugging_face:
            item = self.get_batch_texts(idx)
            item['labels'] = self.get_batch_labels(idx).long()
            return item
        else:
            return self.get_batch_texts(idx), self.get_batch_labels(idx)
