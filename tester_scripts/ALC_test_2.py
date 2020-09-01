'''
this script will test out to investigate the ratio between synthetic:real(input) data.
The main focus is to see the effects of real data ratio increase/decrease.

'''
import warnings


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras.metrics import categorical_accuracy


from modAL_multicol import *
from modAL_multicol.models import ActiveLearner, Committee
from modAL_multicol.multilabel import avg_score
from modAL_multicol.uncertainty import uncertainty_sampling

import numpy as np
import pandas as pd
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from copy import deepcopy
import random

import time
import smtplib

def to_ordinal(y, num_classes=None, dtype='float32'):
    """Converts a class vector of ordinal values to multi-hot binary class matrix.
    E.g. for use with binary_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_ordinal` converts this into a matrix with as many
    # columns are there are classes, minus one, because the
    # first class is represented by a zero vector.
    > to_ordinal(labels)
    array([[ 0.,  0.],
           [ 1.,  1.],
           [ 1.,  0.],
           [ 1.,  1.],
           [ 0.,  0.]], dtype=float32)
    ```
    """
    y = np.array(y, dtype='int')
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

def keras_model(neurons):
    model = Sequential()
    model.add(Dense(units=neurons[0], input_dim=5, activation='relu'))
    if neurons[1] > 0:
        model.add(Dense(units=neurons[1], activation='relu')) # both number of hidden layers AND the number of neurons were determined by https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw    
    model.add(Dense(units=20, activation='sigmoid')) 
    model.compile(loss=binary_crossentropy, optimizer='adam', metrics=[categorical_accuracy])
    return model

def load_AL_models(sc_x, cpy_xpool, cpy_ypool, n_initial, neurons):
    n_members = 2 # initializing number of Committee members
    learner_list = list()
    
    for member_idx in range(n_members):
        train_idx = np.random.choice(range(cpy_xpool.shape[0]), size=n_initial, replace=False)
        X_train, y_train = cpy_xpool[train_idx], cpy_ypool[train_idx]

        cpy_xpool, cpy_ypool = np.delete(cpy_xpool, train_idx, axis=0), np.delete(cpy_ypool, train_idx, axis=0)
        learner = ActiveLearner(
            estimator = keras_model(neurons),
            X_training = X_train, y_training = y_train,
            query_strategy = avg_score
        )

        # not using scaler will make the accuracy be 10% (bad)
        # when the initial synthetic data was not given, then the accuracy can actually go up... 
        # but it can reveal the fact that the accuracy was measured by 
        # how much of the AI can replicate the simulated rater...

        # should the user study focus on replicating the human 'only?'
        # if we have 15 participants, should it only model these 15 people?
        # how do we come up with a generalized solution?
        # what if the human-solutions don't converge?

        # what to test:
        # 1. increase number of neurons
        # 2. try modifying the offset function. then calculate the absolute value of the testing function...


        learner_list.append(learner)

    return Committee(learner_list=learner_list, given_classes = np.array([1,2,3,4,5]))

def give_rating(value):
    d, s, mw, p, hearing_group = value

    if (d > 6000):
        d = 1
    elif (4000 < d <= 6000):
        d = 2
    elif (3000 < d <= 4000):
        d = 3
    elif (2000 < d <= 3000):
        d = 4
    elif (d <= 2000):
        d = 5
    
    if (240 < s) or (s <= 80):
        s = 1
    elif (220 < s <= 240) or (80 < s <= 100):
        s = 2
    elif (200 < s <= 220) or (100 < s <= 120):
        s = 3
    elif (180 < s <= 200) or (120 < s <= 140):
        s = 4
    elif (140 < s <= 180): # ~160wpm
        s = 5
    
    if (6 <= mw): # 6+
        mw = 1
    elif (3 < mw <= 5): # 4,5
        mw = 2
    elif (2 < mw <= 3): # 3
        mw = 3
    elif (0 < mw <= 2): # 1,2
        mw = 4
    elif (mw == 0): # 0
        mw = 5
    
    if (p == 1):
        p = 1
    elif (p == 0):
        p = 5
    return [d, s, mw, p]

def toBinrating(ratings):
    arr = np.zeros(shape=(1, 20)) # in the shape of multiple columns, padd with zeros
    for c in [0,1,2,3]:
        tmp_i = c*5 + ratings[c] - 1 if ratings[c] > 0 else c*5 - 1
        arr[0, tmp_i] = 1
    return arr

def toRating(binary_rate):
    preds = np.split(binary_rate, 4)
    return list(map(lambda x: np.argmax(x), preds))
    
def offset(pred1, pred2):
    '''
        calculate the offset between two predicted ratings,
        then return the absolute value of the difference
        [a,b,c,d], [e,f,g,h] -> abs([a-e, b-f, c-g, d-h])
    '''
    tmp = []
    for i in range(len(pred1)):
        tmp.append(abs(pred1[i] - pred2[i]))
    return tmp

def test_accuracy(com_learner, sc_x):
    data_size = 1000
    accuracy_result = ""

    ## generate 100 random data 
    r_delay = np.random.randint(16000, size=data_size)
    r_wpm = np.random.randint(300, size=data_size)
    missing_words, paraphrasing = [], [] 
    for i in range(data_size):    # because it cannot have a 100% and a missing word..
        paraphrasing_value = random.randint(0, 1) # paraphrasing distribution should reflect sample analysis too. but work on this later.        
        paraphrasing.append(paraphrasing_value)

        if paraphrasing_value > 0:         # 1 = Paraphrased -> some missing words.            
            mw = random.randint(1, 25) # round to integer value.
            missing_words.append(mw)
        else:                              # 0 = Verbatim -> no missing words.
            missing_words.append(0)
    r_paraphrasing = np.asarray(paraphrasing)
    r_missing_words = np.asarray(missing_words)
    hearinggroup = np.full((data_size,1), 1)
    c = np.column_stack((r_delay, r_wpm)) #first two columns, then
    c = np.column_stack((c, r_missing_words))
    c = np.column_stack((c, r_paraphrasing))
    c = np.column_stack((c, hearinggroup))
    np.random.shuffle(c) # shuffle the order in rows   
    
    new_c = sc_x.transform(c)

    # now, let's have the solutions for these
    # get predictions
    offbyone = 0
    counter = 0
    for i in range(len(new_c)):
        tr = toRating(toBinrating(give_rating(c[i]))[0])
        pred = com_learner.predict(new_c[i].reshape(1, -1))
        t_os = offset(pred, tr)
        
        testpr = "{}\tOffset:{}\tSol:{} ==> Pred:{}\n".format(c[i], t_os, tr, pred)
        # accuracy_result += testpr
        if t_os == [0,0,0,0]:
            counter += 1
        else:
            q,w,e,r = t_os
            if q < 2 and w < 2 and e < 2 and r < 2: # let's see how many of them are off by one, for all four factors
                offbyone += 1
                    
    
    summary = "\nGA= {:.2f}%\nRA= {:.2f}%".format(
        counter/data_size*100, (offbyone+counter)/data_size*100
        )
    
    accuracy_result += summary
    print(summary)
    
    return (accuracy_result, summary)

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    """ 


        D = 5 => 0 0 0 0 1
        S = 3 => 0 0 1 0 0
        MW = 4 => 0 0 0 1 0
        PF = 1 => 1 0 0 0 0

        The AI will take an input size of five [D, S, MW, PF, hearinggroup]
        The output form, could be in two types:
            1. by using regression, e.g., prediction [4.3, 3.0, 2.2, 1.4]
            2. by using classification, e.t., prediction [4, 3, 2, 1] # we claim the 'label' be the rating.

            2.a 20 labels, [ 0, 0, 0, 0, 1 | 0 0 1 0 0 | 0 0 0 1 0 | 1 0 0 0 0 ]

                [ 0, 0.2, 0.3, 0.1, 0.4 | ...  ]


                This way, the AI internally predict the relationship between factors.
                -> This is what I originally thought.

            2.b 20 labels, [ 1 1 1 1 1 | 1 1 1 0 0 | 1 1 1 1 0 | 1 0 0 0 0 ]

        D = 5 => 1 1 1 1 1
        S = 3 => 1 1 1 0 0
        MW = 4 => 1 1 1 1 0
        PF = 1 => 1 0 0 0 0

                The AI will have a hint of the relationship between two nominal items.

    """
    
    # print(toRating(np.array([1,0,0,0,0, 0,0,0,0,1, 1,0,0,0,0, 0,0,0,0,1]))) # 0,4,0,4
    # print(toBinrating([1,5,1,5])) # [[1. 0. 0. 0. 0.  0. 0. 0. 0. 1.  1. 0. 0. 0. 0.  0. 0. 0. 0. 1.]]

    outstring = []
    emailstr = []
    n_initial_list = [300] #, 450, 700] #, 1200, 2700, 5700] # testing 50%, 40%, 30%, 20%, 10%, 5% real data    
    vepoch_list = [100]
    neuron_list = [(20, 20)]
    
    real_data_size = 5  # 15 participants and 20 questions per person ~ static value, 300 query-answer pair to be learned.

    datasize = 20000

    dataset = np.array(pd.read_csv("E:/Documents/GitHub/user_study_2/website/home/static/deaf_{}_used.csv".format(datasize), header=None, sep=','))
    # sc_x = StandardScaler()
    sc_x = RobustScaler()
    X_pool = sc_x.fit_transform(dataset[:,:5])
    Y = pd.DataFrame(dataset[:, 5:])
        
    encoded_Y = to_ordinal(Y)
    Y_pool = encoded_Y.reshape([np.size(Y,0), 20])
    cpy_xpool, cpy_ypool = deepcopy(X_pool), deepcopy(Y_pool)

    for version in range(len(n_initial_list)):        
        n_initial = n_initial_list[version]

        train_idx1 = np.random.choice(range(cpy_xpool.shape[0]), size=n_initial, replace=False)
        X_train1, y_train1 = cpy_xpool[train_idx1], cpy_ypool[train_idx1]
        cpy_xpool, cpy_ypool = np.delete(cpy_xpool, train_idx1, axis=0), np.delete(cpy_ypool, train_idx1, axis=0)
        train_idx2 = np.random.choice(range(cpy_xpool.shape[0]), size=n_initial, replace=False)
        X_train2, y_train2 = cpy_xpool[train_idx2], cpy_ypool[train_idx2]
        
        cpy_xpool, cpy_ypool = deepcopy(X_pool), deepcopy(Y_pool)
        for neuron_version in range(len(neuron_list)):
            for epochversion in range(len(vepoch_list)):
                X_pool = cpy_xpool
                
                tic = time.clock()
                init_pr = "|={}====== Neuronset {} \n===== N_INITIAL: {} with epoch {} ============|".format(
                    datasize,
                    neuron_list[neuron_version], n_initial, vepoch_list[epochversion]
                    )
                print(init_pr)
                outstring.append(init_pr)
                emailstr.append(init_pr)
                test_X = []

                neurons = neuron_list[neuron_version]
                # com_learner = load_AL_models(sc_x, cpy_xpool, cpy_ypool, n_initial, neurons)
                learner_list = list()
                learner1 = ActiveLearner(estimator = keras_model(neurons), X_training = X_train1, y_training = y_train1, query_strategy = avg_score)
                learner_list.append(learner1)
                learner2 = ActiveLearner(estimator = keras_model(neurons), X_training = X_train2, y_training = y_train2, query_strategy = avg_score)
                learner_list.append(learner2)
                com_learner = Committee(learner_list=learner_list, given_classes = np.array([1,2,3,4,5]))

                sum_counter_1 = 0
                for i in range(real_data_size): 
                    query_idx, q = com_learner.query(X_pool)
                    queried_x = sc_x.inverse_transform(q)

                    if i%50==0:
                        print(i, queried_x)
                    outstring.append("++ " + str(i) + str(queried_x))

                    original_prediction = com_learner.predict(q) # in range of 0-4
                    rating = toBinrating(give_rating(queried_x[0])) # in range of 1-5
                    teach_rate = toRating(rating[0])
                    oss = offset(original_prediction, teach_rate)

                    # print(sum(oss))

                    if oss == 0:
                        test_X.append((q, teach_rate))
                        sum_counter_1 += 1
                    else:
                        com_learner.teach(q, rating, epochs=vepoch_list[epochversion], verbose=0)
                            
                        new_prediction = com_learner.predict(q)
                        os = offset(new_prediction, teach_rate)
                        opr = "!! after learning Original Pred: {}, offset: {}, Taught Rate: {} - New Pred: {}, so far:".format(original_prediction, os, teach_rate, new_prediction)
                        print(opr)
                        # outstring.append(opr)
                        test_X.append((q, teach_rate))
                        
                    # X_pool = np.delete(X_pool, query_idx, 0)
                toc = time.clock()
                finpr = "It Took: {}\nNow, testing...N={}".format(toc-tic, len(test_X))
                print(finpr)
                outstring.append(finpr)
                
                sum_counter_2 = 0
                for xq, tr in test_X:
                    new_p = com_learner.predict(xq)
                    t_os = offset(new_p, tr)
                    testpr = "{} Offset:{} Sol:{} ==> Pred:{}".format(sc_x.inverse_transform(xq), t_os, tr, new_p)
                    # print(testpr)
                    # outstring.append(testpr)
                    if sum(t_os)==0:
                        sum_counter_2 += 1
                summary_str = "{} out of {}\n{:.2f}->{:.2f}\n = +{:.2f}%".format(sum_counter_2, len(test_X), sum_counter_1/300*100, sum_counter_2/300*100, (sum_counter_2-sum_counter_1)/300*100)
                print(summary_str)
                outstring.append(summary_str)
                emailstr.append(summary_str)

                # test the prediction accuracy
                accuracy_result, ta_summary = test_accuracy(com_learner, sc_x)
                outstring.append(accuracy_result)
                emailstr.append(ta_summary)

            # outfile = open("{}{}({}x{}).txt".format('no_synthdata', neuron_list[neuron_version], n_initial_list[version], vepoch_list[epochversion]),"w")
            # outfile.writelines(l + "\n" for l in outstring)
            # outfile.close() 

        # # Gmail Sign In
        # gmail_sender = 'mfandyou@gmail.com'
        # gmail_passwd = 'vkdnjtprtm'
        # # smtplib module send mail
        # server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.ehlo()
        # server.starttls()
        # server.login(gmail_sender, gmail_passwd)
        # TO = ["sixjuwel@gmail.com"] # must be a list
        # SUBJECT = "Job's done!"
        # TEXT = "" 
        # for i in emailstr:
        #     TEXT += i + "\n"
        # # Prepare actual message
        # message = "Subject: {}\n{}".format(SUBJECT, TEXT)
        # # Send the mail
        # try:
        #     server.sendmail(gmail_sender, TO, message)
        #     print ('email sent')
        # except:
        #     print ('error sending mail')
        # server.quit()





