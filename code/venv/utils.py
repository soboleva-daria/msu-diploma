from pymystem3 import Mystem
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score


class Utility(object):
    @staticmethod
    def set_stem():
        Utility.stem = Mystem()

    @staticmethod
    def sentence_splitter(text):
        sentences = sent_tokenize(text)
        new_sentences = []
        for sentence in sentences:
            if (len(new_sentences) == 0) or ((len(sentence) > 5) and sentence[0] != ')'):
                new_sentences.append(sentence)
            else:
                new_sentences[-1] += sentence
        return new_sentences

    @staticmethod
    def set_stop_words():
        Utility.stop_words = stopwords.words('russian')

    @staticmethod
    def set_interrogative_pronouns():
        Utility.interrogative_pronouns = {'кто', 'что', 'какой', 'чей', 'который', 'сколько', 'когда', 'где', 'куда', 'как', 'откуда', 'почемy', 'зачем'}

    @staticmethod
    def filter_by_question_sentence_words_intersect(question, sentences):
        if not hasattr(Utility, 'stop_words'):
            Utility.set_stop_words()

        question_raw = set(word_tokenize(question)).difference(punctuation).difference(Utility.stop_words)
        filtered = []
        for sentence_number, sentence in enumerate(sentences):
            sentence_raw = set(word_tokenize(sentence)).difference(punctuation).difference(Utility.stop_words)
            if len(question_raw.intersection(sentence_raw)):
                filtered.append(False)
            else:
                filtered.append(True)
        return filtered

    @staticmethod
    def calculate_metric(y_true, y_pred):
        print('Precision: {}'.format(precision_score(y_true, y_pred)))
        print('Recall: {}'.format(recall_score(y_true, y_pred)))
        print('F1-score: {}'.format(f1_score(y_true, y_pred)))

    @staticmethod
    def sentence_to_word(sentences):
        sentences_in_words = list()
        for sentence in sentences:
            sentences_in_words.append(sentence.split())
        return sentences_in_words

    @staticmethod
    def train_test_split(df, test_size=0.3):
        np.random.seed(0)
        test_idxs = np.random.choice(df.index.unique(), size=int(test_size * (df.index.nunique())), replace=False)
        train_idxs = np.setdiff1d(df.index, test_idxs)
        return train_idxs, test_idxs