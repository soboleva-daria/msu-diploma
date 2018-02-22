from whoosh.query import *
from whoosh.qparser import QueryParser
from whoosh import qparser
from whoosh.scoring import BM25F, TF_IDF, Frequency

from question import Question

from copy import copy


class Answer(object):
    def __init__(self, question):
        self.__answer_str = None

    @property
    def question(self):
        return copy(self.__question)

    @property
    def doc_ids(self):
        return copy(self.__doc_ids)

    @doc_ids.setter
    def doc_ids(self, doc_ids):
        self.__doc_ids = copy(doc_ids)

    @property
    def sentence_ids(self):
        return copy(self.__sentence_ids)

    @sentence_ids.setter
    def sentence_ids(self, sentence_ids):
        self.__sentence_ids = copy(sentence_ids)

    @property
    def answer_str(self):
        return copy(self.__answer_str)

    @answer_str.setter
    def answer_str(self, answer_str):
        self.__answer_str = copy(answer_str)


class UpdateAnswerProperties(object):
    def __init__(self, answer):
        self.__answer = answer
