from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    with open(infile, "r") as f:
        lines = f.read().splitlines()
    
    sentences = [line.split("\t")[0] for line in lines]
    labels = [line.split("\t")[-1] for line in lines]


    examples: List[SentimentExample] = []
    for s, l in zip(sentences, labels):
        if l.isdecimal():
            examples.append(SentimentExample(s.split(" "), int(l)))
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    added_words = []
    vocab: Dict[str, int] = {}
    for sentiment_example in examples:
        for word in sentiment_example.words:
            if word not in added_words:
                vocab[word] = len(added_words)
                added_words.append(word)
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow = torch.zeros(len(vocab))

    for word in text:
        if word in vocab.keys():
            bow[vocab[word]] += 1
    
    if binary:
        bow = torch.where(bow > 0, True, False)

    return bow
