#!/usr/bin/python3.7
import collections
import itertools
from typing import *

import numpy as np
import pandas as pd

import nltk
try:
    from nltk.corpus import stopwords
except ImportError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

train_path = "assignment_3_data/reviews/train.txt"
test_path = "assignment_3_data/reviews/test.txt"

stopwords_english = set(stopwords.words('english'))

Review = Tuple[List[str], int]

def review_words_iterator(path: str) -> Iterable[Review]:
    with open(path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 2:
                yield [word.lower() for word in parts[:-1]], int(parts[-1])

def review_sentence_iterator(path: str) -> Iterable[Tuple[str, int]]:
    for review_words, rating in review_words_iterator(path):
        yield " ".join(review_words), rating

def find_word_counts(reviews: Iterable[Review]) -> collections.Counter:
    return collections.Counter(itertools.chain(*(words for words, _ in reviews)))

def without_stopwords(reviews: Iterable[Review]) -> Iterable[Review]:
    for words, rating in reviews:
        yield list(filter(lambda w: w not in stopwords_english, words)), rating

"""
a) (Data processing): First create a list of all reviews and their categories.
You can create a list of tuples where the first item is the review and the second
item is the sentiment, i.e., '0' or '1'.
"""
all_reviews = list(review_sentence_iterator(train_path))
all_review_words = list(review_words_iterator(train_path)) 
"""
b) (Tokenization): Extract all the words from the reviews.
You can either use the string split() method directly or use the following method from NLTK.
"""

word_counts = find_word_counts(all_review_words)
all_words = list(word_counts.keys())
# print(all_words)

"""
Calculate the number of occurence of each word in the entire corpus and report the 10 most common tokens.
You can use the following method from NLTK.
"""
print(word_counts.most_common(10))

"""
c) (Stop words removal): Remove all the stop words from the reviews
and re-calculate the number of occurence of each word in the entire corpus.
"""
reviews_without_stopwords = list(without_stopwords(all_review_words))
word_counts_without_stopwords = find_word_counts(reviews_without_stopwords)
print(word_counts_without_stopwords.most_common(10))


"""
d) (Stemmization): Normalize the reviews by stemming and re-calculate the number
of occurence of each word in the entire corpus.
You can use the following function from NLTK.
"""
from nltk.stem.snowball import EnglishStemmer 
stemmer = EnglishStemmer()

def stemmed(reviews: Iterable[Review]) -> Iterable[Review]:
    for words, rating in reviews:
        yield [stemmer.stem(word) for word in words], rating

stemmed_reviews = list(stemmed(reviews_without_stopwords))
stemmed_word_counts = find_word_counts(stemmed_reviews)
print(stemmed_word_counts.most_common(10))

"""
e) (BOW): Create 1-hot encoding and represent each review as Bag of Words (BOW). 
"""

# First, defining some types to make things easier to read:
ReviewBagOfWords = Tuple[Dict[int, int], int]
ReviewNGram = Tuple[List[Tuple[str, ...]], int]

def vocabulary(reviews: Iterable, size=None) -> Dict[str, int]:
    word_counts = find_word_counts(reviews)
    return {
        word: i
        for i, (word, count) in enumerate(word_counts.most_common(n=size))
    }

def bag_of_words(reviews: Iterable[ReviewNGram], vocab: Dict[Any, int]) -> Iterable[ReviewBagOfWords]:
    for words, rating in reviews:
        word_ids = [vocab[word] for word in words if word in vocab]
        counts = Counter(word_ids)
        
        for word_id in word_ids:
            if word_id not in counts:
                counts[word_id] = 0

        yield counts, rating


from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.util import ngrams as nltk_ngrams


def review_ngrams(path: str, n: int) -> Iterable[ReviewNGram]:
    reviews = review_words_iterator(path)
    reviews_nostopwords = without_stopwords(reviews)
    stemmed_reviews = stemmed(reviews_nostopwords)
    for words, rating in stemmed_reviews:
        yield list(nltk_ngrams(words, n)), rating

# finally, here is the representation for part e)
bow_reviews = review_ngrams(train_path, 1)

"""f) (Train a Classifier): Use reviews in the train folder as "train-set" and use reviews in the test folder as "test-set".
Use Naive Bayes Classifer to train your sentiment predictor. You can use the following code for this purpose.
"""
def accuracy(n: int, vocabulary_size: int = None) -> float:    
    train_vocab = vocabulary(review_ngrams(train_path, n), size=vocabulary_size)
    train_set = list(bag_of_words(review_ngrams(train_path, n), train_vocab))
    test_set =  list(bag_of_words(review_ngrams(test_path,  n), train_vocab))
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    return accuracy

acc_unigrams = accuracy(n=1)
print(f"Accuracy (unigrams): {acc_unigrams}")

"""g) (N-gram): There are different n-grams like unigram, bigram, trigram, etc.
For example, Bigram = Item having two words, such as, very good. BOW and unigram representation as the same.
Extract Bigram features from the reviews and re-train the model with Naive Bayes classifier on the train-set and report accuracy on the test-set.
You can use the following method from NLTK to extract bigrams.
"""

acc_bigrams = accuracy(n=2)
print(f"Accuracy (bigrams): {acc_bigrams}")

"""
h) (Combined features): Represent each review based on the combination of 2000 most frequent unigram and bi-grams.
Re-train the Naive Bayes Classifier and report the accuracy.
"""

acc_2000_unigrams = accuracy(n=1, vocabulary_size=2000)
print(f"Accuracy (2000 most frequent unigrams): {acc_2000_unigrams}")
acc_2000_bigrams = accuracy(n=2, vocabulary_size=2000)
print(f"Accuracy (2000 most frequent bigrams): {acc_2000_bigrams}")

