import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = len(features[0]) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features,labels,delta)
        self.nlabels = len(self.class_priors)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        class_priors = {}
        min= int(torch.min(labels).item())
        max = int(torch.max(labels).item())
        count = labels.numel()

        for i in range(min, max+1):
            class_priors[i] = torch.eq(labels, i).sum()/count

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples. -> apariciones en cada clase
            labels (torch.Tensor): Labels corresponding to each training example. -> label del texto
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        class_word_counts  = {}
        num, den = delta, delta*self.vocab_size
        min, max = int(torch.min(labels).item()), int(torch.max(labels).item())


        for i in range(min, max +1):
            rows = torch.nonzero(labels == i).squeeze() #take only rows with label == i(class == c)
            class_word_counts[i] = (features[rows] + num)/(features[rows].sum() + den)
        return class_word_counts


    def estimate_class_posteriors(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError("Model must be trained before estimating class posteriors.")

        log_posteriors = torch.zeros(len(self.class_priors))

        for c in self.class_priors.keys():
            log_prior = torch.log(self.class_priors[c])
            log_likelihood = (torch.log(self.conditional_probabilities[c]) * feature).sum()
            log_posteriors[c] = log_prior + log_likelihood  

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).
        """
        log_posteriors = self.estimate_class_posteriors(feature)
        return torch.argmax(log_posteriors).item()

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.
        """
        log_posteriors = self.estimate_class_posteriors(feature)
        probs = torch.nn.functional.softmax(log_posteriors, dim=0)  
        return probs