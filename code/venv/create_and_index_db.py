def create_db():
    stem = Mystem()
    for paragraph in tqdm(df.paragraph.unique(), total=df.paragraph.nunique()):
        paragraph_id = ((df[df.paragraph == paragraph].paragraph_id).values[0])

        with open("../data/data_base_origin/{}.txt".format(paragraph_id), 'w') as fout:
            fout.write(paragraph)

        txt_lemm = ''.join(stem.lemmatize(paragraph))
        with open("../data/data_base_lem/{}.txt".format(paragraph_id), 'w') as fout:
            fout.write(txt_lemm)