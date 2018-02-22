import pandas as pd
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT, KEYWORD
from whoosh.query import *
from whoosh import qparser
from whoosh.scoring import BM25F, TF_IDF, Frequency
from fuzzywuzzy import fuzz
from itertools import combinations
import os
from string import punctuation
from tqdm import tqdm
import numpy as np

from utils import Utility


class Question(object):
    def __init__(self, question_str):
        self.question_str = question_str

    def find_question_str_lem(self, question_id=None):
        Utility.set_interrogative_pronouns()
        self.question_str_lem = ' '.join([w for w in ' '.join(Utility.stem.lemmatize(self.question_str)).split() \
                                          if w not in punctuation and w not in ['??'] and w not in Utility.interrogative_pronouns]
                                         )
        #np.save("question_lem/{}_lem".format(question_id), self.question_str_lem)
        #self.question_str_lem = np.load("question_lem/{}_lem.npy".format(question_id)).item()


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

        stem = Utility.stem
        for paragraph in tqdm(df.paragraph.unique(), total=df.paragraph.nunique()):
            paragraph_id = (df[df.paragraph == paragraph].paragraph_id.values[0])

            with open("{}/{}.txt".format(database_origin_dir, paragraph_id), 'w') as fout:
                fout.write(paragraph)

            txt_lemm = ''.join(stem.lemmatize(paragraph))
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

    def find_rel_question_doc_ids(self, question, index_dir="../data/index", question_id=None):
        if not hasattr(self, 'ix'):
            self.ix = open_dir(index_dir)
            self.query_parser = qparser.QueryParser(
                "content",
                schema=self.ix.schema,
                group=qparser.OrGroup
            )

        if not hasattr(question, 'question_str_lem'):
            question.find_question_str_lem(question_id)

        q = self.query_parser.parse(u'{}'.format(question.question_str_lem))
        with self.ix.searcher(weighting=self.search_alg) as searcher:
            results = searcher.search(q)
            doc_ids = [int(doc['title'].split('.')[0]) for doc in results]
        return doc_ids

    @staticmethod
    def create_train_dataset(data_dir='../notebooks/bm25f', database_dir="../data/database_lem"):
        df_dict = {}
        for f in tqdm(os.listdir(data_dir)):
            if f.endswith('.npy'):
                question_id = int(f.split('.')[0])
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
        df_with_list_of_sentences.columns = ['question_id', 'sentence_lem', 'doc_number']
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
            answer_lem = row[1]['answer_lem']
            sentence_lem = row[1]['sentence_lem']
            answer_in_sentence.append(answer_lem in sentence_lem)
        data['answer_in_sentence'] = np.array(answer_in_sentence) * 1
        return data

    @staticmethod
    def get_max_match_sentence(data_row):
        sentences = data_row["sentence_lem"].values
        sentences_in_words = Utility.sentence_to_word(sentences)
        question_in_words = Utility.sentence_to_word([data_row["question_lem"].values[0]])[0]

        max_overlap = None
        max_match_sentence_id = None

        for sentence_id in range(len(sentences_in_words)):
            sentence_words = set(sentences_in_words[sentence_id])
            overlap = len(sentence_words.intersection(question_in_words))
            if max_overlap is None or overlap > max_overlap:
                max_overlap = overlap
                max_match_sentence_id = sentence_id
        return sentences[max_match_sentence_id]

    @staticmethod
    def create_base_features(df):
        if not hasattr(Utility, 'stop_words'):
            Utility.set_stop_words()

        if not hasattr(Utility, 'interrogative_pronouns'):
            Utility.set_interrogative_pronouns()

        stop_words = set(Utility.stop_words)
        interrogative_pronouns = set(Utility.interrogative_pronouns)

        features = {}
        features['sentence_lem_no_punct'] = set(filter(lambda w: w not in punctuation, df.sentence_lem.split()))
        features['sentence_lem_no_stop_words'] = set(filter(lambda w: w not in stop_words, features['sentence_lem_no_punct']))

        features['question_lem_no_punct'] = set(filter(lambda w: w not in punctuation, df.question_lem.split()))
        features['question_lem_no_stop_words'] = set(filter(lambda w: w not in stop_words, features['question_lem_no_punct']))
        features['question_lem_no_interrogative_pronouns'] = set(filter(lambda w: w not in interrogative_pronouns, features['question_lem_no_punct']))

        res = {}
        res['sentence_lem_no_punct_len'] = len(features['sentence_lem_no_punct'])
        res['sentence_lem_no_stop_words_len'] = len(features['sentence_lem_no_stop_words'])

        for i, j in combinations(features, r=2):
            if i.split('_')[0] == j.split('_')[0]:
                continue
            res['{}_{}_intersect_len'.format(i, j)] = len(features[i] - features[j])
            res['{}_{}_fuzz_ratio'.format(i, j)] = fuzz.ratio(features[i], features[j])
            res['{}_{}_fuzz_partial_ratio'.format(i, j)] = fuzz.partial_ratio(features[i], features[j])
            res['{}_{}_fuzz_token_sort_ratio'.format(i, j)] = fuzz.token_sort_ratio(features[i], features[j])
            res['{}_{}_fuzz_token_set_ratio'.format(i, j)] = fuzz.token_set_ratio(features[i], features[j])
            res['{}_{}_jaccard_sim'.format(i, j)] = len(features[i] & features[j]) / len(features[i] | features[j])
        return pd.Series(res)