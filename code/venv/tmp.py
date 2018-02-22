from pymystem3 import Mystem

from copy import copy
import os


class Question(object):
    def __init__(self, question_str):
        self.__question_str = question_str
        self.__question_str_lem = None

    @property
    def question_str(self):
        return self.__question_str

    @property
    def question_str_lem(self):
        return self.__question_str_lem

    @question_str_lem.setter
    def question_str_lem(self, question_str_lem):
        self.__question_str_lem = question_str_lem


class UpdateQuestionProperties(object):
    def __init__(self, question):
        self.__question = copy(question)

    @staticmethod
    def set_stem():
        UpdateQuestionProperties.stem = Mystem()

    def update_question_str_lem(self):
        self.__question.question_str_lem = ' '.join(UpdateQuestionProperties.stem.lemmatize(self.__question.question_str))