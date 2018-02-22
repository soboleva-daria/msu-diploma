import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy
import os
import sys

from qa_system import QuestionAnswerSystem, Question

from tqdm import tqdm

@profile
def tmp():
    df = pd.read_csv('../../data/train_task_b.csv')

    #search_rel_question_doc_alg_str = "BM25F"
    #qa_system = QuestionAnswerSystem(search_rel_question_doc_alg_str)
    #qa_system.add_database_to_index()

    search_rel_question_doc_alg_str = "BM25F"
    if not os.path.exists(search_rel_question_doc_alg_str):
        os.mkdir(search_rel_question_doc_alg_str)
    qa_system = QuestionAnswerSystem(search_rel_question_doc_alg_str)

    accuracy = 0
    errors = {}
    for question_str, paragraph_id, question_id in tqdm(df[['question', 'paragraph_id', 'question_id']].values, total=df.question.nunique()):
        question = Question(question_str)

        rel_doc_ids = qa_system.find_rel_question_doc_ids(question, question_id=question_id, index_dir='../../data/index')
        #np.save('{}/{}.npy'.format(search_rel_question_doc_alg_str, question_id), rel_doc_ids)
        if paragraph_id in rel_doc_ids:
            accuracy += 1
        else:
            errors[question_id] = copy(rel_doc_ids)
    print('{}: Accuracy: {}'.format(search_rel_question_doc_alg_str, accuracy))

tmp()