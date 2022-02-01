import re
import string
import pandas as pd

from icecream import ic
from nltk.corpus import stopwords
from textblob import TextBlob
from tqdm.auto import tqdm

tqdm.pandas()

def clean_data(text_df: pd.DataFrame, to_lower: bool = True,
               remove_mail: bool = True, remove_url: bool = True,
               remove_ref: bool = True, remove_punct: bool = True,
               remove_num: bool = True, correct_spelling: bool = True,
               remove_stopwords: bool = True) -> pd.DataFrame:
    """
    Cleans 'text' column in text_df:
    Args:
        text_df (pd.DataFrame): dataframe with column 'text' containing strings
        to_lower (bool, optional): if True all letters converted to lower. Defaults to True.
        remove_mail (bool, optional): if True removes mail addresses. Defaults to True.
        remove_url (bool, optional): if True removes urls. Defaults to True.
        remove_ref (bool, optional): if True removes twitter references (*@). Defaults to True.
        remove_punct (bool, optional): if True removes punctuation. Defaults to True.
        remove_num (bool, optional): if True removes numbers. Defaults to True.
        correct_spelling (bool, optional): if True corrects typos. Defaults to True.
        remove_stopwords (bool, optional): if True removes stopwords. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with single column 'text' containing cleaned text
    """
    text_df = text_df['text']
    if to_lower:
        ic('To lower')
        text_df = text_df.progress_apply(__to_lower)
    if remove_mail:
        ic('Remove mail')
        text_df = text_df.progress_apply(__remove_email)
    if remove_url:
        ic('Remove urls')
        text_df = text_df.progress_apply(__remove_url)
    if remove_ref:
        ic('Remove Twitter references')
        text_df = text_df.progress_apply(__remove_twitter_references)
    if remove_punct:
        ic('Remove punctuation')
        text_df = text_df.progress_apply(__remove_punctuation)
    if remove_num:
        ic('Remove numbers')
        text_df = text_df.progress_apply(__remove_numbers)
    if correct_spelling:
        ic('Spelling correction')
        text_df = text_df.progress_apply(__spelling_correction)
    if remove_stopwords:
        ic('Remove stopwords')
        text_df = text_df.progress_apply(__remove_stopwords)
    return pd.DataFrame({'text': text_df}) 

def __to_lower(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence with all lower letters
    """
    return sentence.lower()


def __remove_punctuation(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without punctuation
    """
    nopunc = [char for char in sentence if char not in string.punctuation]
    return ''.join(nopunc) 


def __remove_stopwords(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without stopwords
    """
    return ' '.join([word for word in sentence.split() if word.lower() not in stopwords.words('english')])


def __remove_url(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without urls
    """
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , sentence)


def __remove_email(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without mail addresses
    """
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', '', sentence)


def __remove_numbers(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without numbers
    """
    return re.sub('\d+', '', sentence)


def __remove_twitter_references(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence without twitter references(@*)
    """
    return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)" , '',sentence)


def __spelling_correction(sentence: str) -> str:
    """
    Converts all letters in sentence to lower
    Args:
        sentence (str):

    Returns:
        str: sentence with corrected typos
    """
    textBlb = TextBlob(sentence)
    return textBlb.correct()
