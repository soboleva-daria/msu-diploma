from pymystem3 import Mystem


class Question(object):
    def __init__(self, question_str):
        self.question_str = question_str

    @staticmethod
    def set_stem():
        Question.stem = Mystem()

   def update_question_str_lem(self):
        self.question_str_lem = ' '.join(Question.stem.lemmatize(self.question_str))