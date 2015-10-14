from errbot import BotPlugin, botcmd
from random import random
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from spacy.en import English
import gensim
import numpy as np
import os

TWSS_DIR = os.path.dirname(os.path.realpath(__file__))
TWSS_DATA = os.path.join(TWSS_DIR, 'data', 'twss.txt')
NONTWSS_DATA = os.path.join(TWSS_DIR, 'data', 'non_twss.txt')

# This represents the probability that we respond, given something
# said is twss. Lower values will make this bot respond less often.
alpha = 0.3

nlp = English()

def avg_word_vector(doc):
    tokens = nlp(gensim.utils.to_unicode(doc))
    vectors = np.array([t.vector for t in tokens])
    avg_vector = vectors.mean(axis=0)
    if avg_vector.shape != (300,):
        return np.zeros(300)
    return avg_vector

def get_data():
    with open(TWSS_DATA) as f:
        twss = f.readlines()
    with open(NONTWSS_DATA) as f:
        non_twss = f.readlines()

    num_twss = len(twss)

    documents = twss + non_twss
    documents = np.array([avg_word_vector(line) for line in documents])

    targets = np.zeros(len(documents))
    targets[:num_twss] = 1

    shuffled_idx = np.random.permutation(len(documents))

    documents = documents[shuffled_idx]
    targets = targets[shuffled_idx]

    return documents, targets

class TwssBot(BotPlugin):
    """That's what she said responder"""
    min_err_version = '1.6.0'

    def __init__(self, *args, **kwargs):
        super(TwssBot, self).__init__(*args, **kwargs)

        self.model = None
        self.last_message = None

        # The tolerance for considering a message as twss
        self.threshold = 0.8

        self._load_model()

    def _p_twss_response(self, sentence):
        doc = avg_word_vector(sentence)
        return self.model.predict_proba(doc)[0,1]

    def _estimate_threshold(self):
        positive = np.array([
            self._p_twss_response("Does the skin look red and swollen?"),
            self._p_twss_response("just touch it!"),
            self._p_twss_response("put it in"),
            self._p_twss_response("My mother's coming."),
        ]).mean()

        negative = np.array([
            self._p_twss_response("A lot of companies"),
            self._p_twss_response("I'm going to draw a graph"),
        ]).mean()

        diff = abs(positive - negative)

        self.threshold = negative + (diff / 2)
        self.log.info('- threshold %s', self.threshold)

    def _load_model(self):
        try:
            self.model = joblib.load(os.path.join(TWSS_DIR, 'data', 'twss_rf.pkl'))
            return True
        except:
            return False

    @botcmd(split_args_with=None)
    def twss_train(self, mess, args):
        """Create a model for classifying twss text"""

        self.send(mess.frm, "Channelling my inner Michael Scott...", message_type=mess.type)

        X,y = get_data()

        self.log.info('X shape: %s', X.shape)

        self.model = RandomForestClassifier(n_estimators=1000,)
        score = cross_val_score(self.model, X, y, 'log_loss', cv=3, n_jobs=1, verbose=1).mean()
        self.log.info('score: %s', score)

        self.model.fit(X,y)

        joblib.dump(self.model, os.path.join(TWSS_DIR, 'data', 'twss_rf.pkl'))

        self._estimate_threshold()

    @botcmd(split_args_with=None)
    def twss_reload(self, mess, args):
        """Attempt to load a saved model from disk"""
        if self._load_model():
            self.send(mess.frm, "\"You miss 100%% of the shots you don't take ~ Wayne Gretsky\" ~ Michael Scott", message_type=mess.type)

    @botcmd(split_args_with=None)
    def twss_yes(self, mess, args):
        """Save the message that we last responded to as a postive example"""
        if self.last_message is None:
            return

        with open(TWSS_DATA, 'a') as f:
            f.write(self.last_message)
            self.last_message = None

    @botcmd(split_args_with=None)
    def twss_no(self, mess, args):
        """Save the message that we last responded to as a negative example"""
        if self.last_message is None:
            return

        with open(NONTWSS_DATA, 'a') as f:
            f.write(self.last_message)
            self.last_message = None

    @botcmd(split_args_with=None)
    def twss_alpha(self, mess, args):
        """Manually set alpha"""
        self.log.info('set alpha')
        self.log.info(args)
        try:
            alpha = float(args[0])
        except:
            pass

    @botcmd(split_args_with=None)
    def twss_threshold(self, mess, args):
        """Manually set threshold"""
        self.log.info('set threshold')
        self.log.info(args)
        try:
            self.threshold = float(args[0])
        except:
            pass

    def callback_message(self, mess):

        # TODO ignore any messages from self

        if self.model is None:
            return

        p = self._p_twss_response(mess.body)
        self.log.info('-- twss message p: %s, alpha: %s, mess: %s', p, alpha, mess)

        if p > self.threshold and random() < alpha:
            self.last_message = mess.body
            self.send(mess.frm, "That's what she said.", message_type=mess.type)
