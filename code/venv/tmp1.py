from textblob import TextBlob as tb

Utility.set_stem()
Utility.set_interrogative_pronouns()
stem = Utility.stem
interrogative_pronouns = Utility.interrogative_pronouns


def get_interrogative_pronouns(question):
    words_lem = []
    for w in tb(question).words:
        words_lem.append(''.join(stem.lemmatize(w.lower())).replace('\n', ''))
    for w_lem in words_lem:
        if ('год' in words_lem or 'век' in words_lem) and 'какой' in words_lem:
            return 'когда'
        if w_lem in interrogative_pronouns:
            return w_lem


tqdm_pandas(tqdm(total=df.index.nunique()))
df['question_type'] = df.reset_index().groupby('question_id').progress_apply(
    lambda x: get_interrogative_pronouns(x.question.values[0]))


from collections import Counter
Utility.set_stop_words()
Utility.set_morph()
stop_words = Utility.stop_words + ['—']
morph = Utility.morph

def get_most_frequent_words_ner(bloblist):
    words = []
    for blob in bloblist:
        words.extend(list(filter(lambda w: w not in stop_words, tb(blob.lower()).words)))
    most_common_words = [t[0] for t in Counter(words).most_common(10)]
    tags = set()
    for word in most_common_words:
        tags.add(Utility.get_ner(word))
    return tags

tqdm_pandas(tqdm(total=df.question_type.nunique()))
res = df.groupby('question_type').progress_apply(lambda x: get_most_frequent_words_ner(x['answer_lem'])).reset_index()

res.columns = [
    'question_type',
    'most_frequent_{}_ner_by_question_type'.format('answer_lem'),
]

df = pd.merge(df.reset_index(), res, how='left', on='question_type').set_index('question_id')