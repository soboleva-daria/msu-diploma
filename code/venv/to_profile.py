import pandas as pd

from nltk.corpus import stopwords
from pymystem3 import Mystem
from whoosh.qparser import QueryParser
from whoosh import qparser
from string import punctuation
from whoosh.query import *

import os
from tqdm import tqdm

from question import Question
from whoosh.index import open_dir

df = pd.read_csv('../../data/train_task_b.csv')

db_index_dir = "../../data/index"
ix = open_dir(db_index_dir)

class UpdateQuestionProperties(object):
    def __init__(self, question):
        self.__question = question

    @staticmethod
    def set_stem():
        UpdateQuestionProperties.stem = Mystem()

    @staticmethod
    def set_db_index(db_index):
        UpdateQuestionProperties.db_index = db_index

    @staticmethod
    def set_qparser():
        UpdateQuestionProperties.qp = QueryParser(
            "content",
            schema=UpdateQuestionProperties.db_index.schema,
            group=qparser.OrGroup
        )

    @profile
    def update_question_str_lem(self):
        self.__question.question_str_lem = ''.join([w for w in UpdateQuestionProperties.stem.lemmatize(self.__question.question_str) if w not in punctuation])

    @profile
    def update_rel_doc_ids(self):
        q = UpdateQuestionProperties.qp.parse(u'{}'.format(self.__question.question_str_lem))
        rel_doc_ids = []
        with UpdateQuestionProperties.db_index.searcher() as searcher:
            results = searcher.search(q)
            for doc in results:
                rel_doc_ids.append(doc['title'])
        return rel_doc_ids

UpdateQuestionProperties.set_db_index(ix)
UpdateQuestionProperties.set_stem()
UpdateQuestionProperties.set_qparser()

@profile
def f():
    accuracy = 0
    errors = {}
    for paragraph_id, question, question_id in tqdm(df[['paragraph_id', 'question', 'question_id']].values,
                                                    total=df.shape[0]):
        if question_id not in [45825, 2604, 23232]:
            q = Question(
                question_str=question
            )
            updater = UpdateQuestionProperties(question=q)
            updater.update_question_str_lem()

            res = updater.update_rel_doc_ids()
            if len(res):
                doc_match = int(res[0].split('.')[0])

                if (doc_match == paragraph_id):
                    accuracy += 1
                else:
                    errors[question_id] = doc_match
            else:
                errors[question_id] = doc_match

    accuracy /= df.question.nunique()

f()