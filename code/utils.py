from pymystem3 import Mystem
#from polyglot.text import Text
import gensim
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from textblob import TextBlob as tb
from natasha import DatesExtractor
from collections import Counter
import pymorphy2
import pandas as pd
import numpy as np
import math
from tqdm import tqdm, tqdm_pandas
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine
import os
from copy import copy
from joblib import Parallel, delayed
import multiprocessing


class Utility(object):

    @staticmethod
    def set_stem():
        Utility.stem = Mystem()

    @staticmethod
    def set_morph():
        Utility.morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def set_dates_extractor():
        Utility.dates_extractor = DatesExtractor()
   
    @staticmethod
    def set_stop_words():
        Utility.stop_words = stopwords.words('russian')

    @staticmethod
    def set_interrogative_pronouns():
        Utility.interrogative_pronouns = {'кто', 'что', 'какой', 'чей', 'который', 'сколько', 'когда', 'где', 'куда', 'как', 'откуда', 'почемy', 'зачем'}

    @staticmethod
    def set_w2v():
        # Load Google's pre-trained Word2Vec model.
        #model = gensim.models.KeyedVectors.load_word2vec_format('ruwikiruscorpora_upos_skipgram_300_2_2018.vec')  

        #w2v = {}
        #for i in model.vocab:
        #    w2v[i.split('_')[0]] = model[i]
        Utility.w2v = np.load('w2v.npy').item()

    @staticmethod
    def lemmatize(text):
        if not hasattr(Utility, 'stem'):
            Utility.set_stem()
        stem = Utility.stem
        text_lem = ' '.join([''.join(stem.lemmatize(w.lower())).replace('\n', '') for w in word_tokenize(text)])
        return text_lem

    @staticmethod
    def lemmatize_question(text):
        if not hasattr(Utility, 'interrogative_pronouns'):
            Utility.set_interrogative_pronouns()

        if not hasattr(Utility, 'stem'):
            Utility.set_stem()

        stem = Utility.stem
        interrogative_pronouns = Utility.interrogative_pronouns
        return ' '.join(list(filter(lambda w: w not in interrogative_pronouns, [''.join(stem.lemmatize(w.lower())).replace('\n', '') for w in word_tokenize(text) if w not in punctuation])))

    @staticmethod
    def sentence_splitter(text):
        return [str(sentence) for sentence in Text(text, hint_language_code='ru').sentences]

    @staticmethod
    def filter_by_question_sentence_words_intersect(question, sentences):
        if not hasattr(Utility, 'stop_words'):
            Utility.set_stop_words()

        question_raw = set(question.split()).difference(punctuation).difference(Utility.stop_words)
        filtered = []
        for sentence_number, sentence in enumerate(sentences):
            sentence_raw = set(word_tokenize(sentence)).difference(punctuation).difference(Utility.stop_words)
            if len(question_raw.intersection(sentence_raw)):
                filtered.append(False)
            else:
                filtered.append(True)
        return filtered

    @staticmethod
    def train_test_split(df, test_size=0.3):
        np.random.seed(0)
        test_idxs = np.random.choice(df.index.unique(), size=int(test_size * (df.index.nunique())), replace=False)
        train_idxs = np.setdiff1d(df.index, test_idxs)
        return train_idxs, test_idxs

    @staticmethod
    def calc_idf(word, idfs, corpus_size):
        return math.log(corpus_size - idfs[word] + 0.5) - math.log(idfs[word] + 0.5)

    @staticmethod
    def calc_bm25f_score(query_words, sent_words, idfs):
        PARAM_K1 = 1.2
        PARAM_B = 0.75
        counter = Counter(sent_words)
        sent_len = len(sent_words)
        score = 0
        for new_word in set(query_words):
            if new_word in idfs:
                score += (Utility.calc_idf(new_word, idfs, idfs['_corpus_size_']) * counter[new_word] * (PARAM_K1 + 1)) \
                         / (counter[new_word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * sent_len / idfs['_avgdl_']))
        return score

    @staticmethod
    def calc_tf_idf_score(query_words, sent_words, idfs):
        counter = Counter(sent_words)
        score = 0
        for new_word in set(query_words):
            if new_word in idfs:
                score += (Utility.calc_idf(new_word, idfs, idfs['_corpus_size_']) * counter[new_word])
        return score

    @staticmethod
    def stats(question, sentences, idfs):
        tb_sentences = [tb(sentence.lower()) for sentence in sentences]
        question_in_words = set(tb(question.lower()).words)
        len_question_in_words = len(question_in_words)

        unique_word_count_scores = []
        unique_word_percent_scores = []
        tf_idf_scores = []
        bm25f_scores = []
        sentence_len = []
        for sentence_id in range(len(tb_sentences)):
            sentence_in_words = tb_sentences[sentence_id].words
            sentence_len.append(len(sentence_in_words))
            unique_word_count_scores.append(len(question_in_words.intersection(sentence_in_words)))
            unique_word_percent_scores.append(unique_word_count_scores[-1] / len_question_in_words)
            bm25f_scores.append(Utility.calc_bm25f_score(question_in_words, sentence_in_words, idfs))
            tf_idf_scores.append(Utility.calc_tf_idf_score(question_in_words, sentence_in_words, idfs))

        return unique_word_count_scores, unique_word_percent_scores, sentence_len, bm25f_scores, tf_idf_scores

    @staticmethod
    def get_answer_by_score(data, col_with_score):
        return roc_auc_score(data.answer_in_sentence, data[col_with_score])

    @staticmethod
    def get_word_ner(word):
        if word.replace('.', '').replace(',', '').isdigit():
            return 'I-NUM'
        entities = Text(word, hint_language_code='ru').entities
        if len(entities):
            return entities[0].tag

    @staticmethod
    def dump_sentences_ners(sentences):
        ners = []
        for sentence in sentences:
            ners.append(Utility.get_sentence_ners(sentence))
        s = pd.Series([sentences, ners])
        return pd.DataFrame.from_items(zip(s.index, s.values))
	
    @staticmethod
    def applyParallel(dfGrouped, func):
        retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in tqdm(dfGrouped, total=len(dfGrouped)))
        return pd.concat(retLst)

    @staticmethod
    def get_question_type(df):
        if not hasattr(Utility, 'interrogative_pronouns'):
            Utility.set_interrogative_pronouns()
        if not hasattr(Utility, 'morph'):
            Utility.set_morph()
        if not hasattr(Utility, 'stop_words'):
            Utility.set_stop_words()
        interrogative_pronouns = Utility.interrogative_pronouns
        morph = Utility.morph
        stop_words = Utility.stop_words

        question_id = df.index.values[0]
        question = df.question.values[0]
    
        question_interrogative_pronoun = ''
        for w in tb(question).words:
            w_lem = morph.parse(w)[0].normal_form
            if len(question_interrogative_pronoun) and (w_lem not in stop_words):
                df.loc[question_id, 'question_type'] = '{} {}'.format(question_interrogative_pronoun, w_lem)
                return df
            if w_lem in interrogative_pronouns:
                question_interrogative_pronoun = w_lem
        df.loc[question_id, 'question_type'] = question_interrogative_pronoun
        return df

    @staticmethod
    def get_sentence_ners(df):
        if not hasattr(Utility, 'morph'):
            Utility.set_morph()
        if not hasattr(Utility, 'dates_extractor'):
            Utility.set_dates_extractor()
        morph = Utility.morph
        dates_extractor = Utility.dates_extractor

        sentences = df.sentence
        question_id = df.question_id
        sentences_ners = []
        for sentence in sentences:  
             ners = set()
         
             dates_matches = dates_extractor(sentence)
             dates = [token.value for date in dates_matches for token in date.tokens]
             if len(dates):
                 ners.add('Date')
         
             for w in tb(sentence).words:
                 tag = morph.parse(w)[0].tag
                 if ('Name' in tag) or ('Surn' in tag) or ('Patr' in tag): 
                     ners.add('Per')   
                 elif 'Geox' in tag:
                     ners.add('Geox')
                 elif 'Orgn' in tag:
                     ners.add('Orgn')
                 elif ('ROMN' in tag) or (w.split('-')[0].isdigit()) and (len(w.split('-')) > 1):
                     ners.add('Date')
                 elif (w not in dates) and (('NUMR' in tag) or ('NUMB' in tag)):
                     ners.add('Num')
             
             sentences_ners.append(copy(ners))
         
        s = pd.Series([question_id, sentences, sentences_ners])
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_sentence_ners_indicators(df):
        for ner in df.sentence_ners:
             df[ner] = 1
        return df
   
    @staticmethod
    def get_ners_counts_by_question_type(df, ners):
        for ner in ners:	
            df['{}_local'.format(ner)] = df[ner].sum()
        return df

    @staticmethod
    def get_most_freq_ner_question_type(df, ners):
        df['question_type_ner'] = ners[np.argmax(df[['{}_local'.format(ner) for ner in ners]].values[0])]
        return df

    @staticmethod
    def get_sentence_ner_question_type_indicator(df):
        question_type_ner = df.question_type_ner.values[0]
        result = []
        for sentence_ners in df.sentence_ners:
            if question_type_ner in sentence_ners:
                result.append(1)
            else:
                result.append(0)
        s = pd.Series([df.question_id, df.sentence, result])
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_sentence_ner_question_type_interactions(x, combs):
        question_type_ner = x.question_type_ner.values[0]
        result = {}
        for sentence_ners in x.sentence_ners:
            for comb in combs:
                q, s = comb[0], comb[1]
                indicator = 0
                if (q == question_type_ner) and (s in sentence_ners) or (s == question_type_ner) and (q in sentence_ners):
                    indicator = 1
                key = '{}_{}'.format(q, s)
                if key not in result:
                    result[key] = [indicator]
                else:
                    result[key].append(indicator)
        s = pd.Series([
            x.question_id,
            x.sentence,
        ] + [result['{}_{}'.format(comb[0], comb[1])] for comb in combs]
        )
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_question_sentence_word2vec_cosine_dist(x):
        if not hasattr(Utility, 'w2v'):
            Utility.set_w2v()
        w2v = Utility.w2v

        sentence_words = x.sentence_lem.split(' ')
        question_words = x.question_lem.split(' ')
    
        sentence_vec = np.zeros_like(w2v['xxxx'])
        count = 0
        for w in sentence_words:
            if w in w2v:
                sentence_vec += w2v[w]
                count += 1
        if count:
            sentence_vec /= count
    
        question_vec = np.zeros_like(w2v['xxxx'])
        count = 0
        for w in question_words:
            if w in w2v: 
                question_vec += w2v[w]
                count += 1
        if count:
           question_vec /= count
    
        x['question_sentence_word2vec_cosine_dist'] = cosine(sentence_vec, question_vec)
        return x
