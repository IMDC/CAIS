import itertools
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib.staticfiles.templatetags.staticfiles import static

from numpy import loadtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

import modAL_multicol
from modAL_multicol.models import ActiveLearner, Committee
from modAL_multicol.multilabel import avg_score
from modAL_multicol.uncertainty import uncertainty_sampling
from home.multi_encoder import MultiColumnLabelEncoder

from .models import Blobby, Response, Question, AnswerRadio


warnings.simplefilter(action='ignore', category=FutureWarning)
# import modAL
# from modAL.models import ActiveLearner, Committee
# from modAL.uncertainty import uncertainty_sampling

class ActiveLearningClient:

    def keras_model(self):
        """
        This function compiles and returns a Keras model.
        Should be passed to KerasClassifier in the Keras scikit-learn API.
        """
        print("KERAS MODEL GENERATION")
        model = Sequential()
        model.add(Dense(units=10, input_dim=5, activation='relu'))
        model.add(Dense(units=20, activation='relu'))
        model.add(Dense(units=20, activation='sigmoid')) 
        # compile keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_AL_models(self, cpy_xpool, cpy_ypool):
        n_members = 2 # initializing number of Committee members
        learner_list = list()
        n_initial = 700  # number of initial training data ~ this determines the ratio between user_model vs. human input to inquiry

        if not Blobby.objects.exists(): # participant number == 1
            # below for loop would only be launched for the very first participant...
            print("FIRST PARTICIPANT!")
            for member_idx in range(n_members):
                train_idx = np.random.choice(range(cpy_xpool.shape[0]), size=n_initial, replace=False)
                X_train, y_train = cpy_xpool[train_idx], cpy_ypool[train_idx]
                cpy_xpool, cpy_ypool = np.delete(cpy_xpool, train_idx, axis=0), np.delete(cpy_ypool, train_idx, axis=0)
                learner = ActiveLearner(
                    estimator = self.keras_model(),
                    X_training = X_train, y_training = y_train,
                    query_strategy = avg_score
                )
                learner_list.append(learner)
        else: # participant number > 1
            dir_flag = settings.BASE_DIR
            orig_urls = [dir_flag+"/originalfirst.h5", dir_flag+"/originalsecond.h5"]
            mod_urls = [dir_flag+"/modifiedfirst.h5", dir_flag+"/modifiedsecond.h5"]

            for member_idx in range(n_members):
                model_url = orig_urls[member_idx] if Path(orig_urls[member_idx]).is_file() else mod_urls[member_idx]
                print("\tMODEL_MOD at {}\n".format(model_url))

                train_idx = np.random.choice(range(cpy_xpool.shape[0]), size=n_initial, replace=False)            
                X_train, y_train = cpy_xpool[train_idx], cpy_ypool[train_idx]
                cpy_xpool, cpy_ypool = np.delete(cpy_xpool, train_idx, axis=0), np.delete(cpy_ypool, train_idx, axis=0)
                try:
                    model = keras.models.load_model(model_url)  # load the classifier
                except Exception as e:
                    print("Error in alc.py while loading Keras models", e)
                    return
                learner = ActiveLearner(
                    estimator = model,
                    query_strategy = avg_score
                )
                learner_list.append(learner)
        return Committee(learner_list=learner_list, given_classes = np.array([1,2,3,4,5]))

    def __init__(self):
        self.data_prep()
        cpy_xpool, cpy_ypool = deepcopy(self.X_pool), deepcopy(self.Y_pool)
        self.learner = self.load_AL_models(cpy_xpool, cpy_ypool)
        
    def set_data(self, dataset):
        self.sc_x = StandardScaler()
        self.X_pool = self.sc_x.fit_transform(dataset[:, :5])
        Y = pd.DataFrame(dataset[:, 5:])
        encoded_Y = to_categorical(MultiColumnLabelEncoder().fit_transform(Y)) # one hot encoding
        self.Y_pool = encoded_Y.reshape([np.size(Y,0), 20])

    def data_prep(self):
        """ Loads data from given SDT csv file. Returns:
            model: The passive model learned from the Signal Detection Theory user model data.
            sc_x: StandardScaler for X
            X_pool: Input dataset, fit transformed using sc_x
            Y: Output dataset
        """
        csv_url = settings.STATICFILES_DIRS[0] + "/um_data_deaf.csv" # default, But MUST BE CHANGED for the actual learning.
        try:
            df = pd.read_csv(csv_url, header=None, sep=',')
            dataset = np.array(df)
            self.set_data(dataset)
        except Exception as e:
            print(e)
            return

    def get_hearing_group(self):
        q = Question.objects.filter(text="1. What statement best describes your relationship to the Deaf and/or Hard of Hearing Communities?")[0]
        deaf_or_hoh = AnswerRadio.objects.filter(question=q).last().body
        deaf_or_deafend, hoh = ('I am Deafened', 'I identify as Deaf'), 'I am Hard of Hearing'        
        dir_flag = settings.STATICFILES_DIRS[0]
        if deaf_or_hoh in deaf_or_deafend:
            csv_url = dir_flag + "/um_data_deaf.csv"
        elif hoh == deaf_or_hoh:
            csv_url = dir_flag + "/um_data_hoh.csv"
        else: # either option 1 or 5, we load all files
            csv_url = dir_flag + "/c_um_data100000.csv"
        print("HEARING GROUP:{}, loading:{}".format(deaf_or_hoh, csv_url))
        try:
            df = pd.read_csv(csv_url, header=None, sep=',')
            dataset = np.array(df)
        except Exception as e:
            print(e)
            return
        self.set_data(dataset)

    def make_preds(self):
        query_idx, query_inst = self.learner.query(self.X_pool)
        queried_vals = self.sc_x.inverse_transform(query_inst)
        # get machine prediction to be displayed
        # machine_prediction = self.learner.predict(self.X_pool[query_idx].reshape(1, -1)) # is in range 0-4
        machine_prediction = list(np.array(self.learner.predict(queried_vals)) + 1) # add 1 to show in 1-5 scale
        print("machine prediction:", machine_prediction) # these values are +1 from what's predicted.

        self.test_printing(query_idx, queried_vals[0])
        return (query_idx, machine_prediction, queried_vals[0])

    def train_learner(self, query_idx, ratings):
        q_instance = self.X_pool[query_idx]
        np_ratings = np.zeros(shape=(1, 20)) # in the shape of multiple columns, padd with zeros
        for c in [0,1,2,3]:
            tmp_i = c*5 + ratings[c] - 1
            np_ratings[0, tmp_i] = 1

        print("User ratings:", np_ratings)
        # User ratings: [[1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0.]] 
        # we convert form to (1,20) not (,20)
        # therefore, the 0-4 range for index doesn't really matter because we convert from 1-5 range to 1,20 anyways.
        self.learner.teach(q_instance, np_ratings, epochs=100, verbose=0)
        self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
        return (self.learner, self.X_pool)


    def test_printing(self, query_idx, queried_vals):
        pf_val = "Paraphrased" if queried_vals[3] == 1 else "Verbatim"
        hearing_group = "Deaf" if queried_vals[4] == 1 else "Hard of Hearing"
        tmp_str = "The machine selected index {} with raw values:\n\t" \
            + "Delay of {} ms\n\tSpeed of {} WPM\n\tMissing {} words and is {}.\n" \
            + "Predicting ratings by {}."
        print(tmp_str.format(query_idx, queried_vals[0], queried_vals[1], queried_vals[2], pf_val, hearing_group))