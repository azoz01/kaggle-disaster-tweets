import pandas as pd
import nltk
from nltk.corpus import wordnet


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)


def __convert_nltk_tag_to_wordnet(tag: str) -> str:
    """
    Converts pos tags obtained by nltk.pos_tag
    to form compatible with WordNetLemmatizer

    Args:
        tag (str): tag from nltik_pos_tag

    Returns:
        str: tag in WordNetLemmatizer format
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def __lemmatize_sentence(sentence: str) -> str:
    """
    Lemmatizes all words in sentence

    Args:
        sentence (str):

    Returns:
        str: lemmatized sentence
    """
    lemmatizer = nltk.WordNetLemmatizer()
    result = []
    for word, tag in nltk.pos_tag(sentence):
        pos_tag = __convert_nltk_tag_to_wordnet(tag)
        if pos_tag == '':
            result.append(word)
        else:
            result.append(lemmatizer.lemmatize(word, pos=pos_tag))
    return ' '.join(result)


def lemmatize(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lemmatizes all words in 'text' column in text_df

    Args:
        text_df (pd.DataFrame): dataframe with column 'text' containing strings

    Returns:
        pd.DataFrame: dataframe with 'text' column with lemmatized words
    """
    return text_df.apply(__lemmatize_sentence, axis=1).rename('text')
