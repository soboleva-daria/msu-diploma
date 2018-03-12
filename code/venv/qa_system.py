import pandas as pd
from tqdm import tqdm, tqdm_pandas
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT, KEYWORD
from whoosh.query import *
from whoosh import qparser
from whoosh.scoring import BM25F, TF_IDF, Frequency
from itertools import combinations
from nltk.tokenize import word_tokenize
import os
from string import punctuation
from tqdm import tqdm
import numpy as np

from utils import Utility


class QuestionAnswerSystem(object):
    def __init__(self, search_rel_question_doc_alg_str="BM25F"):
        if search_rel_question_doc_alg_str == "TF_IDF":
            self.search_alg = TF_IDF
        elif search_rel_question_doc_alg_str == "Frequency":
            self.search_alg = Frequency
        else:
            self.search_alg = BM25F

        Utility.set_stem()

    def create_database(self, df, database_origin_dir='../data/database_origin', database_lem_dir='../data/database_lem'
    ):
        if not os.path.exists(database_origin_dir):
            os.mkdir(database_origin_dir)
        if not os.path.exists(database_lem_dir):
            os.mkdir(database_lem_dir)

        for paragraph in tqdm(df.paragraph.unique(), total=df.paragraph.nunique()):
            paragraph_id = (df[df.paragraph == paragraph].paragraph_id.values[0])

            with open("{}/{}.txt".format(database_origin_dir, paragraph_id), 'w') as fout:
                fout.write(paragraph)

            txt_lemm = Utility.lemmatize(paragraph)
            with open("{}/{}.txt".format(database_lem_dir, paragraph_id), 'w') as fout:
                fout.write(txt_lemm)

    def add_database_to_index(self, database_dir="../data/database_lem", index_dir="../data/index"):
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
            schema = Schema(
                title=TEXT(stored=True),
                content=TEXT,
                path=ID(stored=True),
                #tags=KEYWORD
            )
            self.ix = create_in(index_dir, schema)

        self.ix = open_dir(index_dir)
        writer = self.ix.writer()
        files = os.listdir(database_dir)
        for f in tqdm(files, total=len(files)):
            if f.endswith(".txt"):
                with open('{}/{}'.format(database_dir, f), 'r') as fin:
                    writer.add_document(
                        title=u'{}'.format(f),
                        path='{}/{}'.format(database_dir, f),
                        content=fin.read()
                    )
        writer.commit()
        self.query_parser = qparser.QueryParser(
            "content",
            schema=self.ix.schema,
            group=qparser.OrGroup
        )

    def find_rel_question_doc_ids(self, question_str_lem, index_dir="../data/index"):
        if not hasattr(self, 'ix'):
            self.ix = open_dir(index_dir)
            self.query_parser = qparser.QueryParser(
                "content",
                schema=self.ix.schema,
                group=qparser.OrGroup
            )

        q = self.query_parser.parse(u'{}'.format(question_str_lem))
        with self.ix.searcher(weighting=self.search_alg) as searcher:
            results = searcher.search(q)
            doc_ids = [int(doc['title'].split('.')[0]) for doc in results]
        return doc_ids

    @staticmethod
    def create_train_dataset(errors, data_dir='../notebooks/bm25f', database_dir="../data/database_origin"):# polyglot works with origin text to split into sentences
        df_dict = {}
        for f in tqdm(os.listdir(data_dir)):
            question_id = int(f.split('.')[0])
            if f.endswith('.npy') and question_id not in errors:
                res = {}
                for doc_number, doc_id in enumerate(np.load('{}/{}'.format(data_dir, f))):
                    with open("{}/{}.txt".format(database_dir, doc_id)) as fin:
                        res[doc_number] = Utility.sentence_splitter(fin.read())
                df_dict[question_id] = res

        df_with_list_of_docs = pd.DataFrame.from_records(df_dict).T
        df_with_list_of_sentences = pd.DataFrame()
        for col in df_with_list_of_docs.columns:
            df_per_doc = df_with_list_of_docs.apply(lambda x: pd.Series(x[col]), axis=1).stack().reset_index(level=1, drop=True).to_frame()
            df_per_doc['doc_number'] = col
            df_with_list_of_sentences = pd.DataFrame.append(
                df_with_list_of_sentences,
                df_per_doc
            )
        df_with_list_of_sentences = df_with_list_of_sentences.reset_index()
        df_with_list_of_sentences.columns = ['question_id', 'sentence', 'doc_number']
        return df_with_list_of_sentences

    @staticmethod
    def filter_train_dataset(train_df):
        filtered_indices = []
        for question_id in tqdm(train_df.question_id.unique(), total=train_df.question_id.nunique()):
            train_df_part = train_df[train_df.question_id == question_id]
            question = train_df_part.question_lem.values[0]
            sentences = train_df_part.sentence_lem.values
            filtered = Utility.filter_by_question_sentence_words_intersect(question, sentences)
            filtered_indices.extend(np.array(train_df_part.index)[np.array(filtered)])

        train_df_filtered = train_df[~train_df.index.isin(filtered_indices)]
        return train_df_filtered

    @staticmethod
    def create_target(data):
        answer_in_sentence = []
        for row in tqdm(data.iterrows(), total=data.shape[0]):
            answer = row[1]['answer']
            sentence = row[1]['sentence']
            if answer[-1] in punctuation:
                answer = answer[:-1]
            answer_in_sentence.append(answer.lower() in sentence.lower())
        data['answer_in_sentence'] = np.array(answer_in_sentence) * 1
        return data

    @staticmethod
    def get_base_stats(question, sentences, question_lem, sentences_lem, idfs, idfs_lem):
        unique_word_count_scores, unique_word_percent_scores, sentence_len, bm25f_scores, tf_idf_scores = Utility.stats(question, sentences, idfs)
        unique_lem_word_count_scores, unique_lem_word_percent_scores, sentence_lem_len, bm25f_lem_scores, tf_idf_lem_scores = Utility.stats(question_lem, sentences_lem, idfs_lem)

        s = pd.Series([
            unique_word_count_scores,
            unique_lem_word_count_scores,

            unique_word_percent_scores,
            unique_lem_word_percent_scores,

            sentence_len,
            sentence_lem_len,

            bm25f_scores,
            bm25f_lem_scores,

            tf_idf_scores,
            tf_idf_lem_scores,

            sentences,
            sentences_lem,
        ])
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_target_statistic(df, train_idxs, test_idxs, target, n_splits=10):
        df = df.copy()

        tqdm_pandas(tqdm(total=df.index.nunique()))
        df['question_type'] = df.reset_index().groupby('question_id').progress_apply(
            lambda x: Utility.get_interrogative_pronouns(x.question.values[0]))

        df_train = Utility.count_target_statistic_by_folds(df.loc[train_idxs], target, n_splits=n_splits)
        df_test = Utility.count_target_statistic(df.loc[train_idxs], df.loc[test_idxs], target)
        df = pd.concat([df_train, df_test])
        print('target statistic finished...')

        tqdm_pandas(tqdm(total=df.question_id.nunique()))
        speech_type_target_stat = df.groupby('question_id').progress_apply(lambda x: Utility.get_speech_type_target_stat(
            x.question_id.values[0],
            x.sentence_lem,
            x['most_frequent_{}_speech_type_by_question_type'.format(target)].values[0]
        ))
        speech_type_target_stat.columns = ['question_id', 'sentence_lem', 'speech_type_{}_stat'.format(target)]

        tqdm_pandas(tqdm(total=df.question_id.nunique()))
        ner_target_stat = df.groupby('question_id').progress_apply(lambda x: Utility.get_ner_target_stat(
            x.question_id.values[0],
            x.sentence_lem,
            x['most_frequent_{}_ner_by_question_type'.format(target)].values[0]
        ))
        ner_target_stat.columns = ['question_id', 'sentence_lem', 'ner_{}_stat'.format(target)]

        return df, pd.merge(speech_type_target_stat, ner_target_stat, how='left', on=('question_id', 'sentence_lem'))



