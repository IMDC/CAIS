import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.metrics import categorical_accuracy

import modAL_multicol
from modAL_multicol.models import ActiveLearner, Committee
from modAL_multicol.disagreement import vote_entropy_sampling
from modAL_multicol.multilabel import avg_score


def to_ordinal(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    ordinal = np.zeros((n, num_classes - 1), dtype=dtype)
    for i, yi in enumerate(y):
        ordinal[i, :yi] = 1
    output_shape = input_shape + (num_classes - 1,)
    ordinal = np.reshape(ordinal, output_shape)
    return ordinal


def keras_model():
    model = Sequential()
    model.add(Dense(units=20, input_dim=5, activation="relu"))
    model.add(Dense(units=20, activation="relu"))
    model.add(Dense(units=20, activation="sigmoid"))
    model.compile(
        loss=binary_crossentropy, optimizer="adam", metrics=["binary_accuracy"]
    )
    return model


commitee_members = 2
learners = list()

df = pd.read_csv(
    "um_20000.csv",
    header=None,
    sep=",",
)
tmp_dataset = np.array(df)
sc_x = RobustScaler()
cpy_xpool = sc_x.fit_transform(tmp_dataset[:, :5])
Y = pd.DataFrame(tmp_dataset[:, 5:])
encoded_Y = to_ordinal(Y)
cpy_ypool = encoded_Y.reshape([np.size(Y, 0), 20])

for member_idx in range(commitee_members):
    train_idx = np.random.choice(range(cpy_xpool.shape[0]), size=300, replace=False)
    X_train, y_train = cpy_xpool[train_idx], cpy_ypool[train_idx]
    cpy_xpool, cpy_ypool = np.delete(cpy_xpool, train_idx, axis=0), np.delete(
        cpy_ypool, train_idx, axis=0
    )
    learner = ActiveLearner(
        estimator=keras_model(),
        query_strategy=avg_score,
        X_training=X_train,
        y_training=y_train,
    )
    learners.append(learner)

act_model_learner = Committee(
    learner_list=learners, given_classes=np.array([1, 2, 3, 4, 5])
)

for i in range(10):
    query_idx, q_instance = act_model_learner.query(cpy_xpool)
    queried_vals = sc_x.inverse_transform(q_instance)
    machine_prediction = list(np.array(act_model_learner.predict(q_instance)) + 1)  # add 1 to show in 1-5 scale    
    print("machine prediction:", machine_prediction, query_idx, queried_vals[0])

    # now teach one-
    ratings=[1,3,4,2]
    np_ratings = np.zeros(
        shape=(1, 20)
    )  # in the shape of multiple columns, padd with zeros
    for c in [0, 1, 2, 3]:
        tmp_start = c * 5
        tmp_i = c * 5 + ratings[c]
        for w in range(tmp_start, tmp_i):
            np_ratings[0, w] = 1

    # e.g., User ratings [1,3,4,2]:
    # [[1. 0. 0. 0. 0. | 1. 1. 1. 0. 0. | 1. 1. 1. 1. 0. | 1. 1. 0. 0. 0.]]
    # we convert form to (1,20) not (,20)
    # therefore, the 0-4 range for index doesn't really matter because we convert from 1-5 range to 1,20 anyways.
    act_model_learner.teach(q_instance, np_ratings, only_new=True, epochs=100, verbose=0)
    machine_prediction = list(np.array(act_model_learner.predict(q_instance)) + 1)  # add 1 to show in 1-5 scale    
    print("machine prediction:", machine_prediction, query_idx, queried_vals[0])

