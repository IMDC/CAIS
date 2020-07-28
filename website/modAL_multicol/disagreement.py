"""
Disagreement measures and disagreement based query strategies for the Committee model.
"""
from collections import Counter
from typing import Tuple
import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
# from tensorflow.keras.models import Sequential
from keras.models import Sequential

from modAL_multicol.utils.data import modALinput
from modAL_multicol.utils.selection import multi_argmax, shuffled_argmax
from modAL_multicol.models.base import BaseCommittee




def vote_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    #edited

    Calculates the vote entropy for the Committee. 
    First it computes the predictions of X for each learner in the Committee (i.e., committee.vote(X) ), then 
    calculates the probability distribution of the votes. 
    The entropy of this distribution is the vote entropy of the Committee, which is returned.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the vote entropy is to be calculated.
        X: The data for which the vote entropy is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Vote entropy of the Committee for the samples in X.
    """
    n_learners = len(committee)
    try:
        votes = committee.vote(X, **predict_proba_kwargs) # the prediction votes
    except NotFittedError:
        print("Not Fitted Error at vote_entropy")
        return np.zeros(shape=(X.shape[0],))   
    
    entr = np.zeros(shape=(X.shape[0],)) 
    n_output_cols = int(votes.shape[1]/n_learners) #4 output columns    
    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)*n_output_cols)) # 200000, 5*8/2 -> 200000, 20 (5 classes * 4 output columns)        
    for vote_idx, vote in enumerate(votes):
        split_vote = np.split(vote, n_learners)
        multi_vote_counter = list(map(lambda c: Counter(list(map(lambda x: x[c], split_vote))), [0,1,2,3])) # for each column, return the counter of votes
        # put the average vote count in the p_vote, in which has [vote_idx, multi_class_idx]
        # multi_class_idx: size of 20 => 5*4 => [0,1,2,3,4 | 5,6,7,8,9 | 10,11,12,13,14 | 15,16,17,18,19]
        for c in range(n_output_cols):
            mvc_c = multi_vote_counter[c]
            for key in mvc_c:
                key = int(key)
                if mvc_c[key] > 0:
                    multi_class_idx = c*5 + key if c > 0 else key
                    p_vote[vote_idx, multi_class_idx] = mvc_c[key]/n_learners
        entr[vote_idx] = entropy(p_vote[vote_idx]) # entropy is calculated for each data_point (index) of X
    
    # print(split_vote, p_vote[vote_idx], entr[vote_idx])
    return entr

def vote_entropy_sampling(committee: BaseCommittee, X: modALinput,
                          n_instances: int = 1, random_tie_break=False,
                          **disagreement_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Vote entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
         the instances from X chosen to be labelled.
    """
    disagreement = vote_entropy(committee, X, **disagreement_measure_kwargs)
    
    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]

def consensus_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the consensus entropy for the Committee. First it computes the class probabilties of X for each learner
    in the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. The entropy of the consensus probability distribution is the vote entropy of the
    Committee, which is returned.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the consensus entropy is to be calculated.
        X: The data for which the consensus entropy is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Consensus entropy of the Committee for the samples in X.
    """
    try:
        proba = committee.predict_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    entr = np.transpose(entropy(np.transpose(proba)))
    return entr


def KL_max_disagreement(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the max disagreement for the Committee. First it computes the class probabilties of X for each learner in
    the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. Then each learner's class probabilities are compared to the consensus distribution
    in the sense of Kullback-Leibler divergence. The max disagreement for a given sample is the argmax of the KL
    divergences of the learners from the consensus probability.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the max disagreement is to be calculated.
        X: The data for which the max disagreement is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Max disagreement of the Committee for the samples in X.
    """
    try:
        p_vote = committee.vote_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    p_consensus = np.mean(p_vote, axis=1)

    learner_KL_div = np.zeros(shape=(X.shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    return np.max(learner_KL_div, axis=1)



def consensus_entropy_sampling(committee: BaseCommittee, X: modALinput,
                               n_instances: int = 1, random_tie_break=False,
                               **disagreement_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Consensus entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = consensus_entropy(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_disagreement_sampling(committee: BaseCommittee, X: modALinput,
                              n_instances: int = 1, random_tie_break=False,
                              **disagreement_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Maximum disagreement sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
         measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = KL_max_disagreement(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_std_sampling(regressor: BaseEstimator, X: modALinput,
                     n_instances: int = 1,  random_tie_break=False,
                     **predict_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Regressor standard deviation sampling strategy.

    Args:
        regressor: The regressor for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **predict_kwargs: Keyword arguments to be passed to :meth:`predict` of the CommiteeRegressor.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = std.reshape(X.shape[0], )

    if not random_tie_break:
        query_idx = multi_argmax(std, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(std, n_instances=n_instances)

    return query_idx, X[query_idx]
