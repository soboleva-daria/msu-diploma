from pymystem3 import Mystem
from polyglot.text import Text
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from textblob import TextBlob as tb
from collections import Counter
import pymorphy2
import pandas as pd
import numpy as np
import math
from tqdm import tqdm, tqdm_pandas
from sklearn.model_selection import KFold


class Utility(object):

    @staticmethod
    def set_stem():
        Utility.stem = Mystem()

    @staticmethod
    def set_morph():
        Utility.morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def set_stop_words():
        Utility.stop_words = stopwords.words('russian')

    @staticmethod
    def set_interrogative_pronouns():
        Utility.interrogative_pronouns = {'кто', 'что', 'какой', 'чей', 'который', 'сколько', 'когда', 'где', 'куда', 'как', 'откуда', 'почемy', 'зачем'}

    @staticmethod
    def get_interrogative_pronouns(question):
        if not hasattr(Utility, 'interrogative_pronouns'):
            Utility.set_interrogative_pronouns()
        if not hasattr(Utility, 'stem'):
            Utility.set_stem()
        stem = Utility.stem
        interrogative_pronouns = Utility.interrogative_pronouns

        for w in tb(question).words:
            w_lem = ''.join(stem.lemmatize(w.lower())).replace('\n', '')
            if w_lem in interrogative_pronouns:
                return w_lem

    @staticmethod
    def get_speech_part(word):
        if not hasattr(Utility, 'morph'):
            Utility.set_morph()
        return str(Utility.morph.parse(word)[0].tag).split(',')[0]

    @staticmethod
    def get_ner(word):
        if word.replace('.', '').replace(',', '').isdigit():
            return 'I-NUM'
        entities = Text(word, hint_language_code='ru').entities
        if len(entities):
            return entities[0].tag

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
    def get_most_frequent_words_ner_and_speech_part(bloblist):
        if not hasattr(Utility, 'stop_words'):
            Utility.set_stop_words()
        if not hasattr(Utility, 'morph'):
            Utility.set_morph()
        stop_words = Utility.stop_words
        stop_words += ['—']
        morph = Utility.morph

        words = []
        for blob in bloblist:
            words.extend(list(filter(lambda w: w not in stop_words, tb(blob.lower()).words)))
        most_common_words = [t[0] for t in Counter(words).most_common(10)]
        tags = set()
        speech_parts = set()
        for word in most_common_words:
            speech_parts.add(Utility.get_speech_part(word))
            tags.add(Utility.get_ner(word))
        return pd.Series([tags, speech_parts])

    @staticmethod
    def count_target_statistic(df_train, df_test, target):
        tqdm_pandas(tqdm(total=df_train.question_type.nunique()))
        res = df_train.groupby('question_type').progress_apply(lambda x: Utility.get_most_frequent_words_ner_and_speech_part(x[target])).reset_index()
        res.columns = [
            'question_type',
            'most_frequent_{}_ner_by_question_type'.format(target),
            'most_frequent_{}_speech_type_by_question_type'.format(target)
        ]
        return pd.merge(df_test.reset_index(), res, how='left', on='question_type')

    @staticmethod
    def count_target_statistic_by_folds(df_train, target, n_splits):
        res = []
        questions = df_train.index.unique()
        for i, j in tqdm(KFold(n_splits=n_splits, random_state=0).split(questions), total=n_splits):
            df_i, df_j = df_train.loc[questions[i]], df_train.loc[questions[j]]
            res_by_fold = Utility.count_target_statistic(df_i, df_j, target)
            res.append(res_by_fold)
        return pd.concat(res)

    @staticmethod
    def get_speech_type_target_stat(question_id, sentences, speech_types):
        if (type(speech_types) == float) or (len(speech_types) == 0) or (speech_types == set({None})):
            s = pd.Series([question_id, list(sentences), [0] * len(sentences)])
            return pd.DataFrame.from_items(zip(s.index, s.values))

        speech_types = speech_types - set({None})
        speech_type_target_stat = []
        for sentence in sentences:
            have_speech_type = 0
            for word in tb(sentence).words:
                speech_type = Utility.get_speech_part(word)
                if speech_type in speech_types:
                    have_speech_type = 1
                    break

            if have_speech_type:
                speech_type_target_stat.append(1)
            else:
                speech_type_target_stat.append(0)

        s = pd.Series([question_id, list(sentences), speech_type_target_stat])
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_ner_target_stat(question_id, sentences, ners):
        if (type(ners) == float) or (len(ners) == 0) or (ners == set({None})):
            s = pd.Series([question_id, list(sentences), [0] * len(sentences)])
            return pd.DataFrame.from_items(zip(s.index, s.values))

        ners = ners - set({None})
        ner_statistic = []
        for sentence in sentences:
            have_ner = 0
            for word in tb(sentence).words:
                ner = Utility.get_ner(word)
                if ner in ners:
                    have_ner = 1
                    break
            if have_ner:
                ner_statistic.append(1)
            else:
                ner_statistic.append(0)

        s = pd.Series([question_id, list(sentences), ner_statistic])
        return pd.DataFrame.from_items(zip(s.index, s.values))

    @staticmethod
    def get_answer_by_score(data, col_with_score):
        predict = np.array(list(data["sentence_lem"]))[np.argsort(list(data[col_with_score]))[::-1]][0]
        answer = data[data.answer_in_sentence == 1].sentence_lem.values[0]
        return int(answer == predict)